#!/usr/bin/env python3
"""
Run feature engineering pipeline
"""
import os
import sys
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import works
from src.ingestion.data_loader import DataLoader
from src.features.feature_engineer import FraudFeatureEngine

def main():
    load_dotenv()
    
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Load data
    print("\n1️⃣ Loading data...")
    loader = DataLoader()
    df = loader.load_data()
    df = loader.basic_cleaning(df)
    
    # Create features
    print("\n2️⃣ Engineering features...")
    engineer = FraudFeatureEngine()
    df = engineer.create_all_features(df)
    
    # Save
    print("\n3️⃣ Saving processed data...")
    output_path = os.getenv('PROCESSED_DATA_PATH')
    engineer.save_processed_data(df, output_path)
    
    print("\n" + "="*60)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*60)
    print(f"\n📁 Output saved to: {output_path}")
    print(f"📊 Total rows: {len(df):,}")
    print(f"📊 Total features: {len(df.columns)}")
    
    # Show sample
    print("\n📋 Sample of engineered features:")
    sample_cols = ['ip', 'app', 'ip_click_count', 'ip_app_count', 'hour', 'is_night', 'is_attributed']
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].head(10))
    
    print("\n📊 All features created:")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()
