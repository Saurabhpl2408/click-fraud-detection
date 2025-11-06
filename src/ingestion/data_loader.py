import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

class DataLoader:
    def __init__(self, file_path=None):
        self.file_path = file_path or os.getenv('RAW_DATA_PATH')
    
    def load_data(self, nrows=None):
        """Load data from CSV"""
        print(f"📁 Loading data from {self.file_path}")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        df = pd.read_csv(self.file_path, nrows=nrows)
        print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    def basic_cleaning(self, df):
        """Remove obvious bad data"""
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove nulls in critical columns
        critical_cols = ['ip', 'click_time']
        df = df.dropna(subset=critical_cols)
        
        # Convert click_time to datetime
        df['click_time'] = pd.to_datetime(df['click_time'])
        
        removed = initial_rows - len(df)
        print(f"✅ Cleaned: {len(df):,} rows ({removed:,} removed)")
        return df
    
    def get_data_summary(self, df):
        """Print data summary"""
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"Total rows: {len(df):,}")
        print(f"Total columns: {len(df.columns)}")
        print(f"\nFraud rate: {df['is_attributed'].mean()*100:.3f}%")
        print(f"Fraud cases: {df['is_attributed'].sum():,}")
        print(f"Legitimate cases: {(df['is_attributed']==0).sum():,}")
        print("\nColumn types:")
        print(df.dtypes)
        print("\nMissing values:")
        print(df.isnull().sum())
        print("\nFirst 3 rows:")
        print(df.head(3))
        print("="*50 + "\n")

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data()
    df = loader.basic_cleaning(df)
    loader.get_data_summary(df)
