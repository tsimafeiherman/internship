import lightgbm as lgb
import numpy as np
import pandas as pd
import os
from typing import Optional
import mlflow
import mlflow.lightgbm
from datetime import datetime

from sklearn.metrics import root_mean_squared_error

class TrainModel:
    def __init__(self,
        train_df: pd.DataFrame,
        split_month: int = 30,
        objective:str = 'tweedie',
        n_estimators: int = 1000,
        learning_rate: float = 0.01,
        max_depth: int = 12,
        num_leaves: int = 64,
        random_state: int = 42,
        verbosity: int = -1,
        experiment_name: str = "predict_future_sales",
        run_name: Optional[str] = None):
        
        self.train_df = train_df
        self.split_month = split_month
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model = None
        self.cat_cols = ["global_category", "shop_city"]
        
        self.params = {
            'objective': objective,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'random_state': random_state,
            'verbosity': verbosity
        }

    def fit(self):
        train_mask = self.train_df['date_block_num'] < self.split_month
        val_mask = self.train_df['date_block_num'] >= self.split_month

        train_pos_mask = train_mask & (self.train_df['item_cnt_month'] > 0)
        val_pos_mask = val_mask & (self.train_df['item_cnt_month'] > 0)

        X_train = self.train_df[train_pos_mask].drop('item_cnt_month', axis=1)
        y_train = self.train_df[train_pos_mask]['item_cnt_month']

        X_val = self.train_df[val_pos_mask].drop('item_cnt_month', axis=1)
        y_val = self.train_df[val_pos_mask]['item_cnt_month']

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

        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)

        self.X_val = X_val
        self.y_val_clipped = y_val_clipped

        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(run_name=self.run_name):
            
            mlflow.log_params(
                {
                    **self.params,
                    'split_month': self.split_month,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'n_features': X_train.shape[1],
                    'cat_features': self.cat_cols,
                    'target_transform': 'clip_0_20',
                    'train_filter': 'positive_only'
                }
            )
            
            mlflow.set_tags(
                {
                    'model_type': 'LightGBM',
                    'objective': 'tweedie',
                    'data_version': 'v1',
                    'feature_engineering': 'sparse_zero_rows_lags_rolling_price'
                }
            )
            
            self.model = lgb.LGBMRegressor(**self.params)

            self.model.fit(
                X_train, y_train_clipped,
                eval_set=[(X_val, y_val_clipped)],
                eval_metric='rmse',
                categorical_feature=self.cat_cols,
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
            )
            
            rmse = self.evaluate()
            
            mlflow.log_metrics({
                'val_rmse': rmse,
                'best_iteration': self.model.best_iteration_
            })
            
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_df.head(20).to_csv('feature_importance_top20.csv', index=False)
            mlflow.log_artifact('feature_importance_top20.csv')
            os.remove('feature_importance_top20.csv')
            
            mlflow.lightgbm.log_model(
                self.model,
                "model",
                registered_model_name="sales_predictor"
            )
            
            print(f"MLflow run: {mlflow.active_run().info.run_id}")
        
        return self

    def evaluate(self):
        if self.model is None:
            raise ValueError("Сначала вызовите fit()")
        
        y_pred = self.model.predict(self.X_val)
        rmse = root_mean_squared_error(self.y_val_clipped, y_pred)
        print(f'RMSE регрессора (только на положительных clipped): {rmse:.4f}')
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
