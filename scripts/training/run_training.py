# scripts/training/run_training.py
import sys
from pathlib import Path
import pandas as pd

_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.training.preprocessed import FinalData
from scripts.training.regression import TrainModel


class TrainPipeline:
    def __init__(
        self,
        data_dir: str = "data",
        encoder_dir: str = "encoders",
        model_path: str = "models/regressor.txt",
        submission_path: str = "submission.csv",
        last_month: int = 34,
        split_month: int = 30,
        verbose: bool = True
    ):
        self.data_dir = data_dir
        self.encoder_dir = encoder_dir
        self.model_path = model_path
        self.submission_path = submission_path
        self.last_month = last_month
        self.split_month = split_month
        self.verbose = verbose
        
        self.train_df = None
        self.test_df = None
        self.model = None
        self.predictions = None
    
    def prepare_data(self):
        """Загрузка и подготовка данных"""
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
        
        if self.verbose:
            print(f"Train shape: {self.train_df.shape}")
            print(f"Test shape: {self.test_df.shape}")
        
        return self
    
    def train_model(self):
        """Обучение модели"""
        if self.train_df is None:
            raise ValueError("Сначала вызовите prepare_data()")
        
        if self.verbose:
            print("\n" + "=" * 50)
            print("ОБУЧЕНИЕ МОДЕЛИ")
            print("=" * 50)
        
        self.model = TrainModel(self.train_df, split_month=self.split_month)
        self.model.train(evaluate=True, save=False)  # сохраним сами через save()
        
        # Сохраняем в нужный путь
        self.model.save(self.model_path)
        
        return self
    
    def predict(self):
        """Предсказания на тесте"""
        if self.model is None:
            raise ValueError("Сначала вызовите train_model()")
        
        if self.verbose:
            print("\n" + "=" * 50)
            print("ПРЕДСКАЗАНИЯ")
            print("=" * 50)
        
        self.predictions = self.model.predict(self.test_df)
        
        if self.verbose:
            print(f"Predictions shape: {self.predictions.shape}")
            print(f"Mean prediction: {self.predictions.mean():.4f}")
            print(f"Max prediction: {self.predictions.max():.4f}")
        
        return self
    
    def save_submission(self):
        """Сохранение submission файла"""
        if self.predictions is None:
            raise ValueError("Сначала вызовите predict()")
        
        # Создаём submission
        submission = pd.DataFrame({
            'ID': range(len(self.predictions)),  # или загрузить исходный test.csv
            'item_cnt_month': self.predictions
        })
        
        submission.to_csv(self.submission_path, index=False)
        
        if self.verbose:
            print(f"\nSubmission saved to {self.submission_path}")
            print(f"Submission shape: {submission.shape}")
        
        return self
    
    def run(self):
        """Запуск полного пайплайна"""
        self.prepare_data()
        self.train_model()
        self.predict()
        self.save_submission()
        
        if self.verbose:
            print("\n" + "=" * 50)
            print("ПАЙПЛАЙН ЗАВЕРШЁН")
            print("=" * 50)
        
        return self


# Точка входа для консольного запуска
if __name__ == "__main__":
    pipeline = TrainPipeline(
        data_dir="data",
        encoder_dir="encoders",
        model_path="models/regressor.txt",
        submission_path="submission.csv",
        verbose=True
    )
    pipeline.run()