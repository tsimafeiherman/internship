import pandas as pd
from pathlib import Path


def get_data(dir: str) -> set[pd.DataFrame]:

    dir = Path(dir)

    item_categories = pd.read_csv(dir / "item_categories.csv")
    items = pd.read_csv(dir / "items.csv")
    sales_train = pd.read_csv(dir / "sales_train.csv")
    sample_submission = pd.read_csv(dir / "sample_submission.csv")
    shops = pd.read_csv(dir / "shops.csv")
    test = pd.read_csv(dir / "test.csv")

    return item_categories, items, sales_train, shops, sample_submission, test
