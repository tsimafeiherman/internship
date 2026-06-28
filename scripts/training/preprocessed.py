from scripts.preprocessing.pipeline import process
from scripts.preprocessing.feature_engeneering import FeatureEngineer

class FinalData():
    def __init__(self,
        data_dir: str = "data",
        save_report_path: str = "report.json",
        verbose: bool = True,
        encoder_dir: str = "encoder",
        last_month: int = 34):
        
        self.data_dir = data_dir
        self.save_report_path = save_report_path
        self.verbose = verbose
        self.encoder_dir = encoder_dir
        self.last_month = last_month
        
        self.dqc_report = None
        
    def get_final_data(self):
        
        train, (sample_submission, test), dqc_report = process(self.data_dir, self.save_report_path, self.verbose)
        
        self.dqc_report = dqc_report
        
        train_df, test_df = FeatureEngineer.do_feature_engeneering(train, test, self.encoder_dir, self.last_month)
        
        return train_df, test_df