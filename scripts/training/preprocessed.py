import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path

from scripts.preprocessing.pipeline import process
from scripts.preprocessing.feature_engeneering import FeatureEngineer


class FinalData:
    def __init__(
        self,
        data_dir: str = "data",
        save_report_path: str = "report.json",
        verbose: bool = True,
        encoder_dir: str = "encoder",
        last_month: int = 34,
    ):

        self.data_dir = data_dir
        self.save_report_path = save_report_path
        self.verbose = verbose
        self.encoder_dir = encoder_dir
        self.last_month = last_month

        self.dqc_report = None

    def get_final_data(self):

        train, (sample_submission, test), dqc_report = process(
            self.data_dir, self.save_report_path, self.verbose
        )

        self.dqc_report = dqc_report

        train_df, test_df = FeatureEngineer.do_feature_engeneering(
            train, test, self.encoder_dir, self.last_month
        )

        self.train_df = train_df
        self.test_df = test_df

        return train_df, test_df

    def save_artifacts(self, artifacts_dir: str = "artifacts"):
        """Сохраняет все артефакты для инференса"""
        out = Path(artifacts_dir)
        out.mkdir(exist_ok=True)

        if hasattr(self, "model") and self.model is not None:
            self.model.booster_.save_model(str(out / "model.txt"))

        encoder_src = Path(self.encoder_dir) / "ordinal_encoder.pkl"
        if encoder_src.exists():
            joblib.dump(joblib.load(encoder_src), out / "ordinal_encoder.pkl")

        data_dir = Path(self.data_dir)
        for fname in ["shops.csv", "items.csv", "item_categories.csv"]:
            src = data_dir / fname
            if src.exists():
                pd.read_csv(src).to_csv(out / fname, index=False)

        train_df = self.train_df  # это датафрейм после всей обработки
        # shop_stats
        shop_stats = train_df[
            [
                "shop_id",
                "shop_total_sales_all_time",
                "shop_num_active_items",
                "shop_num_categories",
            ]
        ].drop_duplicates("shop_id")
        shop_stats.to_csv(out / "shop_stats.csv", index=False)

        # item_history
        item_history_cols = [
            "item_id",
            "first_month",
            "last_month",
            "month_with_sales",
            "total_sales",
            "avg_sales",
        ]
        item_history = train_df[item_history_cols].drop_duplicates("item_id")
        item_history.to_csv(out / "item_history.csv", index=False)

        # price_stats
        price_stats = train_df[
            [
                "item_id",
                "item_price_global_mean",
                "item_price_global_median",
                "item_price_global_std",
            ]
        ].drop_duplicates("item_id")
        price_stats.to_csv(out / "price_stats.csv", index=False)

        # category_price_stats
        cat_price_stats = train_df[
            [
                "global_category",
                "category_price_global_mean",
                "category_price_global_median",
            ]
        ].drop_duplicates("global_category")
        cat_price_stats.to_csv(out / "category_price_stats.csv", index=False)

        # last_lags (данные за last_month-1)
        last_lags = train_df[train_df["date_block_num"] == self.last_month - 1][
            [
                "shop_id",
                "item_id",
                "lag_1",
                "lag_2",
                "lag_3",
                "lag_12",
                "rolling_mean_3",
                "had_sales_lag1",
            ]
        ]
        last_lags.to_csv(out / "last_lags.csv", index=False)

        # thresholds
        expensive_threshold = train_df["item_price_global_mean"].quantile(0.99)
        cheap_threshold = train_df["item_price_global_mean"].quantile(0.25)
        thresholds = {
            "expensive_threshold": expensive_threshold,
            "cheap_threshold": cheap_threshold,
        }
        with open(out / "thresholds.json", "w") as f:
            json.dump(thresholds, f)

        # last_month
        with open(out / "last_month.json", "w") as f:
            json.dump({"last_month": self.last_month}, f)

        print(f"Artifacts saved to {artifacts_dir}")

    @classmethod
    def load_artifacts(cls, artifacts_dir: str):
        """Загружает артефакты и возвращает объект FinalData, готовый к predict"""
        artifacts_dir = Path(artifacts_dir)
        obj = cls.__new__(cls)
        obj.artifacts_dir = artifacts_dir

        import lightgbm as lgb

        obj.model = lgb.Booster(model_file=str(artifacts_dir / "model.txt"))

        obj.encoder = joblib.load(artifacts_dir / "ordinal_encoder.pkl")

        obj.shops = pd.read_csv(artifacts_dir / "shops.csv")
        obj.items = pd.read_csv(artifacts_dir / "items.csv")
        obj.item_categories = pd.read_csv(artifacts_dir / "item_categories.csv")
        # объединяем items с категориями
        obj.items_full = obj.items.merge(
            obj.item_categories, on="item_category_id", how="left"
        )

        obj.shop_stats = pd.read_csv(artifacts_dir / "shop_stats.csv")
        obj.item_history = pd.read_csv(artifacts_dir / "item_history.csv")
        obj.price_stats = pd.read_csv(artifacts_dir / "price_stats.csv")
        obj.category_price_stats = pd.read_csv(
            artifacts_dir / "category_price_stats.csv"
        )
        obj.last_lags = pd.read_csv(artifacts_dir / "last_lags.csv")

        with open(artifacts_dir / "thresholds.json", "r") as f:
            thresholds = json.load(f)
        obj.expensive_threshold = thresholds["expensive_threshold"]
        obj.cheap_threshold = thresholds["cheap_threshold"]

        with open(artifacts_dir / "last_month.json", "r") as f:
            obj.last_month = json.load(f)["last_month"]

        obj.cat_cols = ["global_category", "shop_city"]

        return obj

    def predict(self, shop_id: int, item_id: int) -> float:
        df = pd.DataFrame([{"shop_id": shop_id, "item_id": item_id, "ID": 0}])

        shop_row = self.shops[self.shops["shop_id"] == shop_id]
        shop_name = (
            shop_row.iloc[0]["shop_name"] if not shop_row.empty else "UNKNOWN_SHOP"
        )
        item_row = self.items_full[self.items_full["item_id"] == item_id]
        if not item_row.empty:
            item_name = item_row.iloc[0]["item_name"]
            item_category_name = item_row.iloc[0]["item_category_name"]
        else:
            item_name = "UNKNOWN_ITEM"
            item_category_name = "UNKNOWN_CATEGORY"

        df["shop_name"] = shop_name
        df["item_name"] = item_name
        df["item_category_name"] = item_category_name

        df["date_block_num"] = self.last_month
        df["month_num"] = (df["date_block_num"] % 12) + 1
        df["is_december"] = (df["month_num"] == 12).astype(int)
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

        keywords = [
            "доставка",
            "пакет",
            "билет",
            "фирменный",
            "сертификат",
            "игромир",
            "подарочный",
        ]
        pattern = "|".join(keywords)
        df["service_or_item"] = (
            df["item_name"]
            .str.lower()
            .str.contains(pattern, regex=True, na=False)
            .astype(int)
        )

        df["global_category"] = (
            df["item_category_name"].str.split("-").str[0].str.strip()
        )
        df["shop_city"] = df["shop_name"].str.split(" ").str[0].str.strip()

        # !!! ПРИВОДИМ ТИПЫ ПЕРЕД МЕРДЖАМИ !!!
        # shop_stats
        self.shop_stats["shop_id"] = self.shop_stats["shop_id"].astype(int)
        # item_history
        self.item_history["item_id"] = self.item_history["item_id"].astype(int)
        # last_lags
        self.last_lags["shop_id"] = self.last_lags["shop_id"].astype(int)
        self.last_lags["item_id"] = self.last_lags["item_id"].astype(int)
        # price_stats
        self.price_stats["item_id"] = self.price_stats["item_id"].astype(int)
        # category_price_stats - приводим global_category к строке
        self.category_price_stats["global_category"] = self.category_price_stats[
            "global_category"
        ].astype(str)

        # Теперь мерджи
        df = df.merge(self.shop_stats, on="shop_id", how="left")
        df = df.merge(self.item_history, on="item_id", how="left")
        df = df.merge(self.last_lags, on=["shop_id", "item_id"], how="left")
        df = df.merge(self.price_stats, on="item_id", how="left")
        df = df.merge(self.category_price_stats, on="global_category", how="left")

        df["price_ratio_to_category"] = (
            df["item_price_global_mean"] / df["category_price_global_mean"]
        )
        df["price_ratio_to_category"] = df["price_ratio_to_category"].fillna(1)
        df["is_expensive"] = (
            df["item_price_global_mean"] > self.expensive_threshold
        ).astype(int)
        df["is_cheap"] = (df["item_price_global_mean"] < self.cheap_threshold).astype(
            int
        )

        df = df.drop(
            columns=["ID", "shop_name", "item_name", "item_category_name"],
            errors="ignore",
        )

        for col in self.cat_cols:
            df[col] = df[col].astype(str)
        encoded = self.encoder.transform(df[self.cat_cols])
        for i, col in enumerate(self.cat_cols):
            df[col] = encoded[:, i]

        df = df.fillna(0)
        pred = self.model.predict(df)[0]
        return float(np.clip(pred, 0, 20))
