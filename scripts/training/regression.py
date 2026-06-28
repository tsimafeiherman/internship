import lightgbm as lgb
import numpy as np
import pandas as pd
import os
from typing import Optional
import mlflow
import mlflow.lightgbm
from datetime import datetime
import optuna

from sklearn.metrics import root_mean_squared_error


class TrainModel:
    def __init__(
        self,
        model,
        train_df: pd.DataFrame,
        params: Optional[dict] = None,
        split_month: int = 30,
        experiment_name: str = "predict_future_sales",
        run_name: Optional[str] = None,
    ):

        self.model = model
        self.params = params or {}

        self.train_df = train_df
        self.split_month = split_month
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.cat_cols = ["global_category", "shop_city"]

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def _prepare_data(self):

        if self.X_train is not None:
            return

        train_mask = self.train_df["date_block_num"] < self.split_month
        val_mask = self.train_df["date_block_num"] >= self.split_month

        train_pos_mask = train_mask & (self.train_df["item_cnt_month"] > 0)
        val_pos_mask = val_mask & (self.train_df["item_cnt_month"] > 0)

        X_train = self.train_df[train_pos_mask].drop("item_cnt_month", axis=1)
        y_train = self.train_df[train_pos_mask]["item_cnt_month"]

        X_val = self.train_df[val_pos_mask].drop("item_cnt_month", axis=1)
        y_val = self.train_df[val_pos_mask]["item_cnt_month"]

        y_train_clipped = np.clip(y_train, 0, 20)
        y_val_clipped = np.clip(y_val, 0, 20)

        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)

        for col in self.cat_cols:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)

            categories, uniques = pd.factorize(X_train[col])
            X_train[col] = categories

            val_encoded = pd.Categorical(X_val[col], categories=uniques).codes
            X_val[col] = np.where(val_encoded == -1, len(uniques), val_encoded)

            X_train[col] = X_train[col].astype(int)
            X_val[col] = X_val[col].astype(int)

        self.X_train = X_train.fillna(0)
        self.y_train = y_train_clipped
        self.X_val = X_val.fillna(0)
        self.y_val = y_val_clipped

    def _objective(
        self,
        trial: optuna.Trial,
        objective: str = "tweedie",
        metric: str = "rmse",
        verobosity: int = -1,
        random_state: int = 42,
        n_estimators: int = 1000,
    ) -> float:

        trial_params = {
            "objective": objective,
            "metric": metric,
            "verbosity": verobosity,
            "random_state": random_state,
            "n_estimators": n_estimators,
            # optimization parametrs
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.05, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 6, 16),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-4, 1e2, log=True
            ),
        }

        model = type(self.model)(**trial_params)
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            eval_metric="rmse",
            categorical_feature=self.cat_cols,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        y_pred = model.predict(self.X_val)
        return root_mean_squared_error(self.y_val, y_pred)

    def optimizer(
        self, n_trials: int = 25, timeout: Optional[int] = None, refit: bool = True
    ):

        self._prepare_data()
        print("=" * 50)
        print("OPTUNA HYPERPARAMETER OPTIMIZATION")
        print(f"Trials: {n_trials}")
        print("=" * 50)

        mlflow.set_experiment(self.experiment_name)

        run_name = self.run_name

        def mlflow_callback(study, trial):
            with mlflow.start_run(
                run_name=f"trial_{trial.number}_{run_name}", nested=True
            ):
                mlflow.log_params(trial.params)
                mlflow.log_metric("val_rmse", trial.value)
                mlflow.set_tags(
                    {"trial_number": trial.number, "state": trial.state.name}
                )

        study = optuna.create_study(
            direction="minimize",
            study_name=f"optuna_{datetime.now().strftime('%Y%m%d_%H%M')}",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        study.optimize(
            self._objective,
            n_trials,
            timeout,
            callbacks=[mlflow_callback],
            show_progress_bar=True,
        )

        self.best_params = study.best_params
        self.best_rmse = study.best_value

        if refit:
            print("\n=== Обучение с лучшими параметрами ===")
            self.model = type(self.model)(**self.best_params)
            self.fit()

        print(f"\nЛучший RMSE: {self.best_rmse:.4f}")
        print(f"Лучшие параметры: {self.best_params}")

        return self.model

    def fit(self):

        self._prepare_data()

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_params(
                {
                    **self.params,
                    "split_month": self.split_month,
                    "train_samples": len(self.X_train),
                    "val_samples": len(self.X_val),
                    "n_features": self.X_train.shape[1],
                    "cat_features": self.cat_cols,
                    "target_transform": "clip_0_20",
                    "train_filter": "positive_only",
                }
            )

            mlflow.set_tags(
                {
                    "model_type": type(self.model).__name__,
                    "data_version": "v1",
                    "feature_engineering": "sparse_zero_rows_lags_rolling_price",
                }
            )

            self.model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                eval_metric="rmse",
                categorical_feature=self.cat_cols,
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
            )

            rmse = self.evaluate()

            mlflow.log_metrics(
                {"val_rmse": rmse, "best_iteration": self.model.best_iteration_}
            )

            importance_df = pd.DataFrame(
                {
                    "feature": self.X_train.columns,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            importance_df.head(20).to_csv("feature_importance_top20.csv", index=False)
            mlflow.log_artifact("feature_importance_top20.csv")
            os.remove("feature_importance_top20.csv")

            mlflow.lightgbm.log_model(
                self.model, "model", registered_model_name="sales_predictor"
            )

            print(f"MLflow run: {mlflow.active_run().info.run_id}")

        return self

    def evaluate(self):
        if self.model is None:
            raise ValueError("Сначала вызовите fit()")

        y_pred = self.model.predict(self.X_val)
        rmse = root_mean_squared_error(self.y_val, y_pred)
        print(f"RMSE регрессора (только на положительных clipped): {rmse:.4f}")
        return rmse

    def predict(self, X_test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Сначала вызовите fit()")

        X_test = X_test.fillna(0)

        for col in self.cat_cols:
            if col in X_test.columns:
                X_test[col] = X_test[col].astype(str)
                categories, uniques = pd.factorize(X_test[col])
                X_test[col] = categories
                X_test[col] = X_test[col].astype(int)

        return np.clip(self.model.predict(X_test), 0, 20)

    def save(self, path: str = "models/regressor.txt"):
        if self.model is None:
            raise ValueError("Сначала вызовите fit()")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.booster_.save_model(path)
        print(f"Модель сохранена в {path}")

    def train(self, evaluate: bool = True, save: bool = True):
        self.fit()
        if evaluate:
            self.evaluate()
        if save:
            self.save()
        return self
