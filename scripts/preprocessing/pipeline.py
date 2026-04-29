import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.preprocessing.load_data import get_data
from scripts.preprocessing.dqc import validate_all
from scripts.preprocessing.etl import clear_all, merge_dataframes
from scripts.preprocessing.dqc_after_etl import validate_after_etl

import json

def process(data_dir: str = "data", save_report_path: str = "report.json", verbose: bool = True) -> set:
    
    item_categories, items, sales_train, shops, sample_submission, test = get_data(data_dir)
    
    (item_categories, items, sales_train, shops, sample_submission, test), dqc_report = validate_all(
        item_categories, items, sales_train, shops, sample_submission, test
    )
    
    item_categories, items, sales_train, shops, sample_submission, test = clear_all(
        item_categories, items, sales_train, shops, sample_submission, test, verbose
    )
    
    train, merge_report = merge_dataframes(
        item_categories, items, sales_train, shops
    )
    
    dqc_report["merge_quality"] = merge_report
    
    def convert(obj):
        if hasattr(obj, "item"):
            return obj.item()
        return obj
    
    with open(save_report_path, 'w', encoding="utf-8") as f:
        json.dump(dqc_report, f, indent=3, default=convert)
    print(f"Report saved: {save_report_path}")

    validate_after_etl(train)
    assert (train["item_price"] >= 0).all()
    assert train.duplicated(subset=["date", "shop_id", "item_id"]).sum() == 0
    
    return train, (sample_submission, test), dqc_report