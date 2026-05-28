import lightgbm as lgb
import numpy as np
import pandas as pd
import os
from sklearn.metrics import root_mean_squared_error
from typing import Optional

class TrainModel:
    def __init__(self, train_df: pd.DataFrame, split_month: int = 30):
        self.train_df = train_df
        self.split_month = split_month
        self.model = None
        self.cat_cols = ["global_category", "shop_city"]

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

        self.model = lgb.LGBMRegressor(
            objective='tweedie', 
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=12,
            num_leaves=64,
            random_state=42,
            verbosity=-1
        )

        self.model.fit(
            X_train, y_train_clipped,
            eval_set=[(X_val, y_val_clipped)],
            eval_metric='rmse',
            categorical_feature=self.cat_cols,
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
        )
        
        self.X_val = X_val
        self.y_val_clipped = y_val_clipped
        
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
