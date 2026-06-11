# scripts/training/run_training.py (финальная версия)
import sys
from pathlib import Path
import pandas as pd
import lightgbm as lgb
from datetime import datetime

_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.training.regression import TrainModel
from scripts.training.preprocessed import FinalData


class TrainPipeline:
    def __init__(
        self,
        data_dir: str = "data",
        encoder_dir: str = "encoders",
        model_path: str = "models/regressor.txt",
        submission_path: str = "submission.csv",
        last_month: int = 34,
        split_month: int = 30,
        use_optuna: bool = False,
        n_trials: int = 25,
        verbose: bool = True
    ):
        self.data_dir = data_dir
        self.encoder_dir = encoder_dir
        self.model_path = model_path
        self.submission_path = submission_path
        self.last_month = last_month
        self.split_month = split_month
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.verbose = verbose
        
        self.train_df = None
        self.test_df = None
        self.model = None
        self.trainer = None
        self.predictions = None
    
    def prepare_data(self):
        if self.verbose:
            print("=" * 50)
            print("ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
            print("=" * 50)
        
        final_data = FinalData(
            data_dir=self.data_dir,
            save_report_path="report.json",
            verbose=self.verbose,
            encoder_dir=self.encoder_dir,
            last_month=self.last_month
        )
        
        self.train_df, self.test_df = final_data.get_final_data()
        self.final_data = final_data
        self.final_data.train_df = self.train_df
        self.final_data.test_df = self.test_df
        
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        
        return self
    
    def train_model(self):
        if self.verbose:
            print("\n" + "=" * 50)
            print("ОБУЧЕНИЕ МОДЕЛИ")
            print("=" * 50)
        
        model = lgb.LGBMRegressor(
            objective='tweedie',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=12,
            num_leaves=64,
            random_state=42,
            verbosity=-1
        )
        
        self.trainer = TrainModel(
            model=model,
            train_df=self.train_df,
            params=model.get_params(),
            split_month=self.split_month,
            experiment_name="predict_future_sales",
            run_name=f"v1_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        
        if self.use_optuna:
            if self.verbose:
                print("Запуск с Optuna оптимизацией...")
            self.trainer.optimizer(n_trials=self.n_trials, refit=True)
        else:
            self.trainer.train(evaluate=True, save=False)
        
        self.trainer.save(self.model_path)  
        self.model = self.trainer.model
        self.final_data.model = self.trainer.model
        self.final_data.save_artifacts("artifacts")
        
        return self
    
    def predict(self):
        if self.verbose:
            print("\n" + "=" * 50)
            print("ПРЕДСКАЗАНИЯ")
            print("=" * 50)
        
        self.predictions = self.trainer.predict(self.test_df)
        
        print(f"Predictions shape: {self.predictions.shape}")
        print(f"Mean prediction: {self.predictions.mean():.4f}")
        print(f"Max prediction: {self.predictions.max():.4f}")
        
        return self
    
    def save_submission(self):
        if self.verbose:
            print("\n" + "=" * 50)
            print("СОХРАНЕНИЕ SUBMISSION")
            print("=" * 50)
        
        submission = pd.DataFrame({
            'ID': range(len(self.predictions)),
            'item_cnt_month': self.predictions
        })
        
        submission.to_csv(self.submission_path, index=False)
        print(f"Saved to {self.submission_path}")
        
        return self
    
    def run(self):
        self.prepare_data()
        self.train_model()
        self.predict()
        self.save_submission()
        
        if self.verbose:
            print("\n" + "=" * 50)
            print("ПАЙПЛАЙН ЗАВЕРШЁН")
            print("=" * 50)
        
        return self


if __name__ == "__main__":
    pipeline = TrainPipeline(use_optuna=True, n_trials=10, verbose=True)
    pipeline.run()