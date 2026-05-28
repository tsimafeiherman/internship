import pandas as pd

class DQC():
    
    @staticmethod
    def validate_categories(item_categories: pd.DataFrame):
        
        report = {}
        report["name"] = "categories"

        expected_columns = {"item_category_id", "item_category_name"}
        assert set(item_categories.columns) == expected_columns

        report["n_rows"] = len(item_categories)
        report["n_missing"] = item_categories.isna().sum().to_dict()

        assert item_categories.item_category_id.notna().all()
        assert item_categories.item_category_name.notna().all()

        assert pd.api.types.is_string_dtype(item_categories.item_category_name)
        assert pd.api.types.is_integer_dtype(item_categories.item_category_id)

        report["duplicate_ids"] = int(item_categories["item_category_id"].duplicated().sum())
        assert item_categories["item_category_id"].is_unique

        report["empty_names"] = int((item_categories.item_category_name.str.strip() == "").sum())
        assert (item_categories.item_category_name.str.strip() != "").all()

        return item_categories, report

    @staticmethod
    def validate_items(items: pd.DataFrame, item_categories: pd.DataFrame):

        report = {}
        report["name"] = "items"

        expected_columns = {'item_name', 'item_id', 'item_category_id'}
        assert set(items.columns) == expected_columns

        report["n_rows"] = len(items)
        report["missing"] = items.isna().sum().to_dict()

        assert pd.api.types.is_string_dtype(items.item_name)
        assert pd.api.types.is_integer_dtype(items.item_id)
        assert pd.api.types.is_integer_dtype(items.item_category_id)

        report["duplicate_item_id"] = int(items["item_id"].duplicated().sum())
        assert items.item_id.is_unique

        report["empty_names"] = int((items.item_name.str.strip() == "").sum())

        # assert items.item_name.notna().all() change from asssert to ifno in report
        assert items.item_id.notna().all()
        assert items.item_category_id.notna().all()

        report["bad_category_fk"] = int((~items.item_category_id.isin(
            item_categories.item_category_id
        )).sum())

        assert items.item_category_id.isin(item_categories.item_category_id).all()

        return items, report

    @staticmethod
    def validate_sales(sales_train: pd.DataFrame, items: pd.DataFrame, shops: pd.DataFrame):

        report = {}
        report["name"] = "sales"

        expected_columns = {'date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day'}
        assert set(sales_train.columns) == expected_columns

        report["n_rows"] = len(sales_train)

        try:
            dates = pd.to_datetime(sales_train.date, dayfirst=True)
            report["unparsed_dates"] = int(dates.isna().sum())
            assert dates.notna().all()
        except Exception as e:
            assert False, str(e)

        report["missing"] = sales_train.isna().sum().to_dict()

        assert pd.api.types.is_integer_dtype(sales_train.date_block_num)
        assert pd.api.types.is_integer_dtype(sales_train.shop_id)
        assert pd.api.types.is_integer_dtype(sales_train.item_id)
        assert pd.api.types.is_float_dtype(sales_train.item_price)
        assert pd.api.types.is_float_dtype(sales_train.item_cnt_day)

        report["negative_price"] = int((sales_train.item_price < 0).sum())
        report["zero_price"] = int((sales_train.item_price == 0).sum())

        # assert sales_train.item_price.notna().all()  change from asssert to ifno in report
        # assert sales_train.item_cnt_day.notna().all()  change from asssert to ifno in report

        bad_items = ~sales_train.item_id.isin(items.item_id)
        bad_shops = ~sales_train.shop_id.isin(shops.shop_id)

        report["bad_items"] = int(bad_items.sum())
        report["bad_shops"] = int(bad_shops.sum())

        assert bad_items.sum() == 0
        assert bad_shops.sum() == 0

        report["duplicate_rows"] = int(sales_train.duplicated().sum())
        report["duplicate_keys"] = int(sales_train.duplicated(
            subset=["date", "shop_id", "item_id"]
        ).sum())

        # assert not sales_train.duplicated(
        #     subset=["date", "shop_id", "item_id"]
        # ).any()

        price_99 = sales_train.item_price.quantile(0.99)
        cnt_99 = sales_train.item_cnt_day.quantile(0.99)
        report["price_outliers"] = int((sales_train.item_price > price_99).sum())
        report["cnt_outliers"] = int((sales_train.item_cnt_day > cnt_99).sum())
        
        items_with_single_block = sales_train.groupby("item_id")["date_block_num"].nunique()
        report["items_with_one_block"] = int((items_with_single_block == 1).sum())

        return sales_train, report   
    
    @staticmethod
    def validate_shops(shops: pd.DataFrame):

        report = {}
        report["name"] = "shops"

        expected_columns = {'shop_name', 'shop_id'}
        assert set(shops.columns) == expected_columns

        report["n_rows"] = len(shops)
        report["missing"] = shops.isna().sum().to_dict()

        assert pd.api.types.is_string_dtype(shops.shop_name)
        assert pd.api.types.is_integer_dtype(shops.shop_id)

        report["empty_names"] = int((shops.shop_name.str.strip() == "").sum())
        report["duplicate_shop_id"] = int(shops.shop_id.duplicated().sum())

        # assert shops.shop_name.notna().all() change from asssert to ifno in report
        assert shops.shop_id.notna().all()
        assert shops.shop_id.is_unique

        return shops, report

    @staticmethod
    def validate_sample_submission(sample_submission: pd.DataFrame):

        report = {}
        report["name"] = "sample_sub"

        expected_columns = {"ID", "item_cnt_month"}
        assert set(sample_submission.columns) == expected_columns

        report["n_rows"] = len(sample_submission)

        assert pd.api.types.is_integer_dtype(sample_submission.ID)
        assert pd.api.types.is_float_dtype(sample_submission.item_cnt_month)

        report["negative_preds"] = int((sample_submission.item_cnt_month < 0).sum())

        assert sample_submission.ID.notna().all()
        # assert sample_submission.item_cnt_month.notna().all() change from asssert to ifno in report

        assert sample_submission.ID.is_unique
        assert (sample_submission.item_cnt_month >= 0).all()

        return sample_submission, report

    @staticmethod
    def validate_test(test: pd.DataFrame, items: pd.DataFrame, shops:pd.DataFrame):

        report = {}
        report["name"] = "test"

        expected_columns = {"ID", "shop_id", "item_id"}
        assert set(test.columns) == expected_columns

        report["n_rows"] = len(test)

        assert pd.api.types.is_integer_dtype(test.ID)
        assert pd.api.types.is_integer_dtype(test.shop_id)
        assert pd.api.types.is_integer_dtype(test.item_id)

        assert test.ID.notna().all()
        assert test.shop_id.notna().all()
        assert test.item_id.notna().all()

        assert test.ID.is_unique

        report["n_shops"] = test.shop_id.nunique()
        report["n_items"] = test.item_id.nunique()
        
        bad_shops = ~test.shop_id.isin(shops.shop_id)
        report["bad_shops"] = int(bad_shops.sum())
        
        bad_items = ~test.item_id.isin(items.item_id)
        report["bad_items"] = int(bad_items.sum())
        
        if report["bad_shops"] > 0:
            print(f"{report['bad_shops']} записей в test с отсутствующими shop_id")
            print(test[bad_shops][["ID", "shop_id", "item_id"]].head())
        
        if report["bad_items"] > 0:
            print(f"{report['bad_items']} записей в test с отсутствующими item_id")
            print(test[bad_items][["ID", "shop_id", "item_id"]].head())
            
        assert report["bad_shops"] == 0, f"Найдены shop_id, которых нет в shops: {test[bad_shops]['shop_id'].unique()}"
        assert report["bad_items"] == 0, f"Найдены item_id, которых нет в items: {test[bad_items]['item_id'].unique()}"

        return test, report

    @staticmethod
    def validate_all(
        item_categories: pd.DataFrame,
        items: pd.DataFrame,
        sales_train: pd.DataFrame,
        shops: pd.DataFrame,
        sample_submission: pd.DataFrame,
        test: pd.DataFrame) -> set[set[pd.DataFrame], dict]:

        item_categories, categories_report = DQC.validate_categories(item_categories)
        items, items_report = DQC.validate_items(items, item_categories)
        shops, shops_report = DQC.validate_shops(shops)
        sales_train, sales_report = DQC.validate_sales(sales_train, items, shops)
        sample_submission, sample_sub_report = DQC.validate_sample_submission(sample_submission)
        test, test_report = DQC.validate_test(test, items, shops)
        
        final_report = {}
        for report in [
            categories_report, items_report, shops_report,
            sales_report, sample_sub_report, test_report
        ]:
            final_report[f"{report['name']}"] = report
        
        return (item_categories, items, sales_train, shops, sample_submission, test), final_report