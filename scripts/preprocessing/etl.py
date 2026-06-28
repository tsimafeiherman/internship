import pandas as pd


class ETL:
    @staticmethod
    def clear_categories(item_categories: pd.DataFrame) -> pd.DataFrame:

        df = item_categories.copy()

        df.item_category_name = df.item_category_name.fillna("UNKNOWN_CATEGORY")
        df.loc[df.item_category_name.str.strip() == "", "item_category_name"] = (
            "UNKNOWN_CATEGORY"
        )

        return df

    @staticmethod
    def clear_items(items: pd.DataFrame) -> pd.DataFrame:

        df = items.copy()

        df.item_name = df.item_name.fillna("UNKNOWN_ITEM")

        df.item_name = df.item_name.str.strip()
        df.item_name = df.item_name.str.replace(
            r"[^a-zA-Zа-яА-Я0-9\s\.\-\(\)]", "", regex=True
        )
        df.item_name = df.item_name.str.replace(r"\s+", " ", regex=True)
        df.item_name = df.item_name.str.replace(r"\s*\(.*?\)\s*", "", regex=True)
        df.item_name = df.item_name.str.replace(r'["\']', "", regex=True)

        name_duplicates = df[df.duplicated(subset="item_name", keep=False)]
        if len(name_duplicates) > 0:
            print(
                f"Обнаружено {len(name_duplicates)} записей с дублирующимися названиями после чистки!"
            )
            print("Примеры дубликатов:")
            print(name_duplicates[["item_id", "item_name"]].head(10))

            duplicate_names = (
                name_duplicates.groupby("item_name").size().reset_index(name="count")
            )
            duplicate_names = duplicate_names[duplicate_names["count"] > 1]
            print(
                f"\nНайдено {len(duplicate_names)} уникальных названий, которые повторяются:"
            )
            print(duplicate_names.head(10))

            print("\nВероятная причина:")
            print(
                "- Разные товары с одинаковым названием после очистки от спецсимволов"
            )

            print("\Что делать:")
            print("- Оставить как есть — это разные товары с одинаковым названием")
            print("- Или добавить в название артикул/ID, если критично")
        else:
            print("Дубликатов названий после чистки не обнаружено")

        return df

    @staticmethod
    def clear_shops(shops: pd.DataFrame) -> pd.DataFrame:

        df = shops.copy()

        df.shop_name = df.shop_name.fillna("UNKNOWN_SHOP")

        df.shop_name = df.shop_name.str.strip()
        df.shop_name = df.shop_name.str.replace(
            r"[^a-zA-Zа-яА-Я0-9\s\.\-\(\)]", "", regex=True
        )
        df.shop_name = df.shop_name.str.replace(r"\s+", " ", regex=True)

        name_duplicates = df[df.duplicated(subset="shop_name", keep=False)]

        if len(name_duplicates) > 0:
            print(
                f"Обнаружено {len(name_duplicates)} записей с дублирующимися названиями магазинов!"
            )
            print(name_duplicates[["shop_id", "shop_name"]].head(10))

        return df

    @staticmethod
    def clear_sample_sub(sample_submission: pd.DataFrame) -> pd.DataFrame:

        df = sample_submission.copy()

        df.item_cnt_month = df.item_cnt_month.fillna(df.item_cnt_month.median())

        return df

    @staticmethod
    def clear_sales(sales: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Очистка данных о продажах.

        Причины решений:
        - Отрицательные цены: возможны из-за возвратов или ошибок ввода.
        Заменяем на медианную цену товара (если нет истории — глобальную медиану).
        - Отрицательное количество: не бывает отрицательных продаж.
        Обрезаем до 0 (clip(lower=0)).
        - Полные дубликаты строк: результат сбоя при записи.
        Удаляем (drop_duplicates), оставляя первую запись.
        - Дубликаты по ключу (date, shop_id, item_id): один товар мог быть пробит
        несколько раз в один день. Суммируем количество (item_cnt_day),
        цену берём первую (обычно одинаковая).
        """

        df = sales.copy()

        df.date = pd.to_datetime(df.date, dayfirst=True)
        ### fill na data

        if verbose:
            print("\n=== ОЧИСТКА sales_train ===")
            print(f"  Исходно строк: {len(df)}")
            print(f"  Отрицательных цен: {(df['item_price'] < 0).sum()}")
            print(f"  Отрицательных количеств: {(df['item_cnt_day'] < 0).sum()}")
            print(f"  Полных дубликатов строк: {df.duplicated().sum()}")

        item_median = df.groupby("item_id").item_price.transform("median")
        global_median = df.item_price.median()
        df.item_price = df.item_price.mask(
            df.item_price < 0, item_median.fillna(global_median)
        )

        df.date_block_num = df.date_block_num.fillna(df.date_block_num.median())

        df.item_cnt_day = df.item_cnt_day.clip(lower=0)

        df = df.drop_duplicates()

        before_agg = len(df)
        df = df.groupby(
            ["date", "date_block_num", "shop_id", "item_id"], as_index=False
        ).agg({"item_price": "first", "item_cnt_day": "sum"})

        if verbose:
            print(f"  После удаления полных дубликатов: {before_agg} строк")
            print(f"  После агрегации по ключу: {len(df)} строк")
            print(f"  Отрицательных цен осталось: {(df['item_price'] < 0).sum()}")
            print(
                f"  Отрицательных количеств осталось: {(df['item_cnt_day'] < 0).sum()}"
            )

        return df

    @staticmethod
    def clear_all(
        item_categories: pd.DataFrame,
        items: pd.DataFrame,
        sales_train: pd.DataFrame,
        shops: pd.DataFrame,
        sample_submission: pd.DataFrame,
        test: pd.DataFrame,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame]:

        item_categories = ETL.clear_categories(item_categories)
        items = ETL.clear_items(items)
        shops = ETL.clear_shops(shops)
        sales_train = ETL.clear_sales(sales_train, verbose)
        sample_submission = ETL.clear_sample_sub(sample_submission)

        return item_categories, items, sales_train, shops, sample_submission, test

    @staticmethod
    def merge_dataframes(
        item_categories: pd.DataFrame,
        items: pd.DataFrame,
        sales_train: pd.DataFrame,
        shops: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict]:

        merged = pd.merge(sales_train, items, on="item_id", how="left")
        merged = pd.merge(merged, item_categories, on="item_category_id", how="left")
        merged = pd.merge(merged, shops, on="shop_id", how="left")

        merge_report = {}

        merge_report["total_rows"] = len(merged)
        merge_report["null_counts"] = merged.isnull().sum().to_dict()
        merge_report["null_columns_count"] = int((merged.isnull().sum() > 0).sum())

        missing_items = sales_train[~sales_train["item_id"].isin(items["item_id"])]
        merge_report["missing_items_count"] = len(missing_items)
        if merge_report["missing_items_count"] > 0:
            merge_report["missing_item_examples"] = (
                missing_items[["item_id", "shop_id", "item_cnt_day"]]
                .head(10)
                .to_dict(orient="records")
            )

        missing_categories = items[
            ~items["item_category_id"].isin(item_categories["item_category_id"])
        ]
        merge_report["missing_categories_count"] = len(missing_categories)
        if merge_report["missing_categories_count"] > 0:
            merge_report["missing_category_examples"] = (
                missing_categories[["item_id", "item_category_id"]]
                .head(10)
                .to_dict(orient="records")
            )

        missing_shops = sales_train[~sales_train["shop_id"].isin(shops["shop_id"])]
        merge_report["missing_shops_count"] = len(missing_shops)
        if merge_report["missing_shops_count"] > 0:
            merge_report["missing_shop_examples"] = (
                missing_shops[["shop_id", "item_id", "item_cnt_day"]]
                .head(10)
                .to_dict(orient="records")
            )

        merge_report["has_issues"] = (
            merge_report["missing_items_count"] > 0
            or merge_report["missing_categories_count"] > 0
            or merge_report["missing_shops_count"] > 0
            or merge_report["null_columns_count"] > 0
        )

        if merge_report["null_columns_count"] > 0:
            print("\n=== Обнаружены пустые значения после мёрджа ===")
            null_cols = {k: v for k, v in merge_report["null_counts"].items() if v > 0}
            print(f"Колонки с пропусками: {null_cols}")

            if merge_report["missing_items_count"] > 0:
                print(
                    f"\nНайдено {merge_report['missing_items_count']} записей с item_id, отсутствующими в items"
                )
                print("Будут заполнены как 'UNKNOWN_ITEM' в clear_items()")

            if merge_report["missing_categories_count"] > 0:
                print(
                    f"\nНайдено {merge_report['missing_categories_count']} товаров с category_id, отсутствующими в item_categories"
                )
                print("Будут заполнены как 'UNKNOWN_CATEGORY' в clear_categories()")

            if merge_report["missing_shops_count"] > 0:
                print(
                    f"\nНайдено {merge_report['missing_shops_count']} записей с shop_id, отсутствующими в shops"
                )
                print("Будут заполнены как 'UNKNOWN_SHOP' в clear_shops()")
        else:
            print("Merge выполнен корректно, пустых значений не появилось")

        return merged, merge_report
