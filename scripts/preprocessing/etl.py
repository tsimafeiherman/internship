import pandas as pd

def clear_categories(item_categories: pd.DataFrame) -> pd.DataFrame:
    
    df = item_categories.copy()
    
    df.item_category_name = df.item_category_name.fillna("UNKNOWN_CATEGORY")
    df.loc[df.item_category_name.str.strip() == "", "item_category_name"] = "UNKNOWN_CATEGORY"
    
    return df

def clear_items(items: pd.DataFrame) -> pd.DataFrame:
    """
        need to fix reg expressions
    """
    
    df = items.copy()
    
    df.item_name = df.item_name.fillna("UNKNOWN_ITEM")
    
    df.item_name = df.item_name.str.strip()
    df.item_name = df.item_name.str.replace(r'[^a-zA-Zа-яА-Я0-9\s\.\-\(\)]', '', regex=True)
    df.item_name = df.item_name.str.replace(r'\s+', ' ', regex=True)
    df.item_name = df.item_name.str.replace(r'\s*\(.*?\)\s*', '', regex=True)
    df.item_name = df.item_name.str.replace(r'["\']', '', regex=True)
    
    return df
    
def clear_shops(shops: pd.DataFrame) -> pd.DataFrame:
    
    df = shops.copy()
    
    df.shop_name = df.shop_name.fillna("UNKNOWN_SHOP")
    
    return df

def clear_sample_sub(sample_submission: pd.DataFrame) -> pd.DataFrame:
    
    df = sample_submission.copy()
    
    df.item_cnt_month = df.item_cnt_month.fillna(df.item_cnt_month.median())
    
    return df

def clear_sales(sales: pd.DataFrame) -> pd.DataFrame:
    
    df = sales.copy()
    
    df.date = pd.to_datetime(df.date, dayfirst=True)
    ### fill na data
    
    item_median = df.groupby("item_id").item_price.transform("median")
    global_median = df.item_price.median()
    df.item_price = df.item_price.mask(df.item_price < 0, item_median.fillna(global_median))
    
    df.date_block_num = df.date_block_num.fillna(df.date_block_num.median())
    
    df.item_cnt_day = df.item_cnt_day.clip(lower=0)
    
    df = df.drop_duplicates()
    
    df = df.groupby(["date", "date_block_num", "shop_id", "item_id"], as_index=False).agg(
        {
            "item_price":"first",
            "item_cnt_day":"sum"
        }
    )

    return df    

def clear_all(
    item_categories: pd.DataFrame,
    items: pd.DataFrame,
    sales_train: pd.DataFrame,
    shops: pd.DataFrame,
    sample_submission: pd.DataFrame,
    test: pd.DataFrame) -> tuple[pd.DataFrame]:
    
    item_categories = clear_categories(item_categories)
    items = clear_items(items)
    shops = clear_shops(shops)
    sales_train = clear_sales(sales_train)
    sample_submission = clear_sample_sub(sample_submission)
    
    return item_categories, items, sales_train, shops, sample_submission, test

def merge_dataframes(
    item_categories: pd.DataFrame,
    items: pd.DataFrame,
    sales_train: pd.DataFrame,
    shops: pd.DataFrame) -> pd.DataFrame:
    
    merged = pd.merge(sales_train, items, on="item_id", how="left")
    merged = pd.merge(merged, item_categories, on="item_category_id", how="left")
    merged = pd.merge(merged, shops, on="shop_id", how="left")
    
    return merged