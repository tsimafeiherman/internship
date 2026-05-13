import pandas as pd

def validate_after_etl(train: pd.DataFrame):
    
    assert (train.item_price >= 0).all(), "Still rows with item_price <= 0 exist"
    assert (train.item_cnt_day >= 0).all(), "Still rows with item_price <= 0 exist"
    assert train.isna().sum().sum() == 0, "Still None cells exist"
    
    expected_columns = {'date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day', 'item_name', "item_category_id", "item_category_name", "shop_name"}
    assert set(train.columns) == expected_columns, "Mismatch in expecting columns"
    
    assert pd.api.types.is_datetime64_any_dtype(train.date)
    assert pd.api.types.is_integer_dtype(train.date_block_num)
    assert pd.api.types.is_integer_dtype(train.shop_id)
    assert pd.api.types.is_integer_dtype(train.item_id)
    assert pd.api.types.is_float_dtype(train.item_price)
    assert pd.api.types.is_float_dtype(train.item_cnt_day)
    assert pd.api.types.is_object_dtype(train.item_name)
    assert pd.api.types.is_integer_dtype(train.item_category_id)
    assert pd.api.types.is_object_dtype(train.item_category_name)
    assert pd.api.types.is_object_dtype(train.shop_name)