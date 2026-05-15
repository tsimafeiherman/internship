import lightgbm as lgb
import numpy as np
import pandas as pd
import os
from sklearn.metrics import root_mean_squared_error
from typing import Optional

from scripts.training.preprocessed import get_final_data

def train_regression(train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None, split_month: int = 30):

    train_mask = train_df['date_block_num'] < split_month
    val_mask = train_df['date_block_num'] >= split_month

    train_pos_mask = train_mask & (train_df['item_cnt_month'] > 0)
    val_pos_mask = val_mask & (train_df['item_cnt_month'] > 0)

    X_train = train_df[train_pos_mask].drop('item_cnt_month', axis=1)
    y_train = train_df[train_pos_mask]['item_cnt_month']

    X_val = train_df[val_pos_mask].drop('item_cnt_month', axis=1)
    y_val = train_df[val_pos_mask]['item_cnt_month']

    y_train_clipped = np.clip(y_train, 0, 20)
    y_val_clipped = np.clip(y_val, 0, 20)

    # y_train_log = np.log1p(y_train_clipped)
    # y_val_log = np.log1p(y_val_clipped)

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    cat_cols = ["global_category", "shop_city"]
    for col in cat_cols:
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

    model = lgb.LGBMRegressor(
        objective='tweedie', 
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=12,
        num_leaves=64,
        random_state=42,
        verbosity=-1
    )

    model.fit(
        X_train, y_train_clipped,
        eval_set=[(X_val, y_val_clipped)],
        eval_metric='rmse',
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
    )

    y_pred = model.predict(X_val)

    # y_pred_log = model.predict(X_val)
    # y_pred = np.expm1(y_pred_log)
    # y_pred_clipped = np.clip(y_pred, 0, 20)

    rmse = root_mean_squared_error(y_val_clipped, y_pred)
    print(f'RMSE регрессора (только на положительных clipped): {rmse:.4f}')

    os.makedirs("models", exist_ok=True)
    model.booster_.save_model("models/regressor.txt")
    
if __name__ == "__main__":
    train_regression()
