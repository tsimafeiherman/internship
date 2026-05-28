from scripts.preprocessing.pipeline import process
from scripts.preprocessing.feature_engeneering import do_feature_engeneering

def get_final_data(
    data_dir: str = "data",
    save_report_path: str = "report.json",
    verbose: bool = True,
    encoder_dir: str = "encoder",
    last_month: int = 34
    ):
    
    train, (sample_submission, test), dqc_report = process(data_dir, save_report_path, verbose)
    
    train_df, test_df = do_feature_engeneering(train, test, encoder_dir, last_month)
    
    return train_df, test_df