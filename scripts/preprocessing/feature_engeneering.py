import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os

class FeatureEngineer():

    @staticmethod
    def add_binary_service_or_item(df: pd.DataFrame) -> pd.DataFrame:
        keywords = ['доставка', 'пакет', 'билет', 'фирменный', 'сертификат', 'игромир', 'подарочный']
        pattern = '|'.join(keywords)
        
        df["service_or_item"] = df["item_name"].str.lower().str.contains(pattern, regex=True, na=False).astype(int)
        print("Successfuly passed service or item part")
        
        return df

    @staticmethod
    def add_global_category_and_city(df: pd.DataFrame):
        
        df["global_category"] = df["item_category_name"].str.split("-").str[0].str.strip()
        df["shop_city"] = df["shop_name"].str.split(" ").str[0].str.strip()
        
        print("Successfuly passed global_category and shop_city part")
        
        return df

    @staticmethod
    def add_date_features(df: pd.DataFrame, is_train=True, last_month=None):
        
        if is_train:
            df["month_num"] = df.date.dt.month
        else:
            if last_month is None:
                raise ValueError("Для теста нужно указать last_month")
            df["date_block_num"] = last_month
            df["month_num"] = (df["date_block_num"] % 12) + 1
        
        df["is_december"] = (df["month_num"] == 12).astype(int)
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
        
        print("Successfuly passed data_features part")
        
        return df

    @staticmethod
    def aggregate_to_monthly(df: pd.DataFrame):
        
        df = df.groupby(['date_block_num', 'shop_id', 'item_id']).agg(
            month_num=('month_num', 'first'),
            item_price_mean=('item_price', 'mean'),
            item_cnt_month=('item_cnt_day', 'sum'),
            # item_category_id=('item_category_id', 'first'),
            global_category=('global_category', 'first'),
            shop_city=('shop_city', 'first'),
            month_sin=('month_sin', 'first'),
            month_cos=('month_cos', 'first'),
            is_december=('is_december', 'first'),
            service_or_item=('service_or_item', 'first'),
            # item_name_length=('item_name_length', 'first'),
        ).reset_index()
        
        print("Successfuly passed month aggregation part")
        
        return df

    @staticmethod
    def add_sparse_zero_rows_fast(df: pd.DataFrame):

        df = df.sort_values(
            ["shop_id", "item_id", "date_block_num"]
        )

        pairs = df.groupby(
            ["shop_id", "item_id"]
        )["date_block_num"].agg(["min", "max"]).reset_index()

        expanded = []

        for row in pairs.itertuples(index=False):

            months = np.arange(row.min, row.max + 1)

            expanded.append(
                pd.DataFrame({
                    "date_block_num": months,
                    "shop_id": row.shop_id,
                    "item_id": row.item_id
                })
            )

        expanded_df = pd.concat(expanded, ignore_index=True)

        merged = expanded_df.merge(
            df,
            on=["date_block_num", "shop_id", "item_id"],
            how="left"
        )

        merged["item_cnt_month"] = merged["item_cnt_month"].fillna(0)

        fill_cols = [
            "month_num",
            "global_category",
            "shop_city",
            "month_sin",
            "month_cos",
            "is_december",
            "service_or_item",
            "item_price_mean"
        ]

        merged = merged.sort_values(
            ["shop_id", "item_id", "date_block_num"]
        )

        merged[fill_cols] = merged.groupby(
            ["shop_id", "item_id"]
        )[fill_cols].ffill().bfill()

        print(f"Original rows: {len(df)}")
        print(f"New rows: {len(merged)}")

        return merged

    @staticmethod
    def add_lags_and_rolling(df: pd.DataFrame):
        df = df.sort_values(["shop_id", "item_id", "date_block_num"])
        
        gp = df.groupby(["shop_id", "item_id"])
        
        for lag in [1, 2, 3, 12]:
            df[f"lag_{lag}"] = gp['item_cnt_month'].shift(lag).fillna(0)
            
        shift_1 = gp['item_cnt_month'].shift(1).fillna(0)
        
        df["rolling_mean_3"] = gp['item_cnt_month'].shift(1).rolling(3, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
        
        df['had_sales_lag1'] = (df['lag_1'] > 0).astype(int)
        
        print("Successfuly passed lags and rolling part")
        return df

    @staticmethod
    def add_item_history_features(df: pd.DataFrame, last_month: int):
        
        ### Тут может быть утечка данных , так как считаем историю до last month
        ### а потмо можем использовать данные для месяца m из месяцев выше
        
        history = df[df['date_block_num'] < last_month]
        
        history_info = history.groupby("item_id").agg(
            first_month=("date_block_num", "min"),
            last_month=("date_block_num", "max"),
            month_with_sales=("date_block_num", "nunique"),
            total_sales=("item_cnt_month", "sum"),
            avg_sales=("item_cnt_month", "mean")
        ).reset_index()
        
        df = df.merge(history_info, on="item_id", how="left")
        
        print("Successfuly passed item history features part")
        
        return df

    @staticmethod
    def add_shop_aggregates(df: pd.DataFrame):
        
        shop_stats = df.groupby("shop_id").agg(
            shop_total_sales_all_time=("item_cnt_month", "sum"),
            shop_num_active_items=("item_id", "nunique"),
            shop_num_categories=("global_category", "nunique")
        ).reset_index()
        
        df = df.merge(shop_stats, on='shop_id', how='left')
        
        print("Successfuly passed shop aggregations part")
        
        return df

    @staticmethod
    def add_global_price_features(df: pd.DataFrame):
        
        price_stats = df.groupby("item_id").agg(
            item_price_global_mean=("item_price_mean", "mean"),
            item_price_global_median=("item_price_mean","median"),
            item_price_global_std=("item_price_mean","std")
        ).reset_index()
        
        category_price_stats = df.groupby("global_category").agg(
            category_price_global_mean=("item_price_mean", "mean"),
            category_price_global_median=("item_price_mean", "median")
        ).reset_index()
        
        df = df.merge(price_stats, on="item_id", how="left")
        df = df.merge(category_price_stats, on="global_category", how="left")
        
        df = df.drop(columns=["item_price_mean"])
        
        df['item_price_global_std'] = df['item_price_global_std'].fillna(0)

        df["price_ratio_to_category"] = (df.item_price_global_mean / df.category_price_global_mean).fillna(1)
        
        expensive_threshold = df['item_price_global_mean'].quantile(0.99)
        cheap_threshold = df['item_price_global_mean'].quantile(0.25)
        
        df["is_expensive"] = (df.item_price_global_mean > expensive_threshold).astype(int)
        df["is_cheap"] = (df.item_price_global_mean < cheap_threshold).astype(int)
        
        print("Successfuly passed global price features part")
        
        return df

    @staticmethod
    def add_target_encodings(df: pd.DataFrame):
        pass

    @staticmethod
    def prepare_test(base_train: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, encoder_dir: str = "encoders", last_month: int = 34):
        
        test = test_df.copy()
        
        test = FeatureEngineer.add_date_features(test, is_train=False, last_month=last_month)
        
        unique_shops = base_train[['shop_id', 'shop_name']].drop_duplicates()
        test = test.merge(unique_shops, on='shop_id', how='left')
        unique_items = base_train[['item_id', 'item_name', "item_category_name"]].drop_duplicates()
        test = test.merge(unique_items, on='item_id', how='left')
        
        test = FeatureEngineer.add_binary_service_or_item(test)
        test = FeatureEngineer.add_global_category_and_city(test)
        
        shop_stats = train_df[['shop_id', 'shop_total_sales_all_time', 'shop_num_active_items', 'shop_num_categories']].drop_duplicates('shop_id')
        test = test.merge(shop_stats, on='shop_id', how='left')
        
        item_history = train_df[['item_id', 'first_month', 'last_month', 'month_with_sales', 'total_sales', 'avg_sales']].drop_duplicates('item_id')
        test = test.merge(item_history, on='item_id', how='left')
        
        last_month_data = train_df[train_df['date_block_num'] == (last_month - 1)][['shop_id', 'item_id', 'lag_1', 'lag_2', 'lag_3', 'lag_12', 'rolling_mean_3', 'had_sales_lag1']]
        test = test.merge(last_month_data, on=['shop_id', 'item_id'], how='left')
        
        item_price_stats = train_df[['item_id', 'item_price_global_mean', 
                                    'item_price_global_median', 'item_price_global_std']].drop_duplicates(subset='item_id')

        category_price_stats = train_df[['global_category', 'category_price_global_mean', 
                                        'category_price_global_median']].drop_duplicates(subset='global_category')
        test['global_category'] = test['global_category'].astype(str)
        category_price_stats['global_category'] = category_price_stats['global_category'].astype(str)
        
        test = test.merge(item_price_stats, on='item_id', how='left')
        test = test.merge(category_price_stats, on='global_category', how='left')
        
        test['price_ratio_to_category'] = test['item_price_global_mean'] / test['category_price_global_mean']
        test['price_ratio_to_category'] = test['price_ratio_to_category'].fillna(1)
        
        expensive_threshold = train_df['item_price_global_mean'].quantile(0.99)
        cheap_threshold = train_df['item_price_global_mean'].quantile(0.25)
        test['is_expensive'] = (test['item_price_global_mean'] > expensive_threshold).astype(int)
        test['is_cheap'] = (test['item_price_global_mean'] < cheap_threshold).astype(int)
        
        test = test.drop(columns=['ID', 'shop_name', 'item_name', 'item_category_name'])
        
        test = FeatureEngineer.encode_categoricals_transform(test, encoder_dir)
        
        test = test.fillna(0)
        
        return test

    @staticmethod
    def encode_categoricals_fit(df: pd.DataFrame, encoder_dir: str = "encoders"):
        cat_cols = ["global_category", "shop_city"]
        
        for col in cat_cols:
            df[col] = df[col].astype(str)
        
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[cat_cols] = encoder.fit_transform(df[cat_cols])
        
        os.makedirs(encoder_dir, exist_ok=True)
        joblib.dump(encoder, f"{encoder_dir}/ordinal_encoder.pkl")

    @staticmethod
    def encode_categoricals_transform(df: pd.DataFrame, encoder_dir: str = "encoders"):
        cat_cols = ["global_category", "shop_city"]
        
        for col in cat_cols:
            df[col] = df[col].astype(str)
        
        encoder = joblib.load(f"{encoder_dir}/ordinal_encoder.pkl")
        df[cat_cols] = encoder.transform(df[cat_cols])
        
        return df

    @staticmethod
    def do_feature_engeneering(train: pd.DataFrame, test:pd.DataFrame, encoder_dir: str = "encoders", last_month: int = 34):
        
        train_df = FeatureEngineer.add_binary_service_or_item(train)
        train_df = FeatureEngineer.add_global_category_and_city(train_df)
        train_df = FeatureEngineer.add_date_features(train_df, is_train=True)
        train_df = FeatureEngineer.aggregate_to_monthly(train_df)
        train_df = FeatureEngineer.add_sparse_zero_rows_fast(train_df)
        train_df = FeatureEngineer.add_lags_and_rolling(train_df)
        train_df = FeatureEngineer.add_item_history_features(train_df, last_month=33)
        train_df = FeatureEngineer.add_shop_aggregates(train_df)
        train_df = FeatureEngineer.add_global_price_features(train_df)
        FeatureEngineer.encode_categoricals_fit(train_df)
        train_df = FeatureEngineer.encode_categoricals_transform(train_df)
        
        test_df = FeatureEngineer.prepare_test(train, train_df, test, encoder_dir=encoder_dir, last_month=last_month)
        
        return train_df, test_df