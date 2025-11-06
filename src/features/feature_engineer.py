import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class FraudFeatureEngine:
    def __init__(self):
        self.feature_names = []
    
    def create_time_features(self, df):
        """Extract time-based features"""
        print("⏰ Creating time features...")
        
        df['hour'] = df['click_time'].dt.hour
        df['day'] = df['click_time'].dt.day
        df['dayofweek'] = df['click_time'].dt.dayofweek
        
        # Suspicious hours (late night/early morning have more fraud)
        df['is_night'] = df['hour'].between(0, 6).astype(int)
        df['is_working_hours'] = df['hour'].between(9, 17).astype(int)
        
        return df
    
    def create_ip_features(self, df):
        """IP-based fraud signals"""
        print("🌐 Creating IP features...")
        
        # Count clicks per IP
        ip_counts = df.groupby('ip').size().reset_index(name='ip_click_count')
        df = df.merge(ip_counts, on='ip', how='left')
        
        # Count unique apps per IP (many apps = suspicious)
        ip_app_count = df.groupby('ip')['app'].nunique().reset_index(name='ip_app_count')
        df = df.merge(ip_app_count, on='ip', how='left')
        
        # Count unique devices per IP
        ip_device_count = df.groupby('ip')['device'].nunique().reset_index(name='ip_device_count')
        df = df.merge(ip_device_count, on='ip', how='left')
        
        # IP fraud rate (if IP has many clicks with low conversion)
        ip_conversion = df.groupby('ip')['is_attributed'].agg(['sum', 'count']).reset_index()
        ip_conversion['ip_conversion_rate'] = ip_conversion['sum'] / ip_conversion['count']
        df = df.merge(ip_conversion[['ip', 'ip_conversion_rate']], on='ip', how='left')
        
        return df
    
    def create_app_features(self, df):
        """App-based features"""
        print("📱 Creating app features...")
        
        # Clicks per app
        app_counts = df.groupby('app').size().reset_index(name='app_click_count')
        df = df.merge(app_counts, on='app', how='left')
        
        # App conversion rate
        app_conversion = df.groupby('app')['is_attributed'].agg(['sum', 'count']).reset_index()
        app_conversion['app_conversion_rate'] = app_conversion['sum'] / app_conversion['count']
        df = df.merge(app_conversion[['app', 'app_conversion_rate']], on='app', how='left')
        
        return df
    
    def create_channel_features(self, df):
        """Channel (ad publisher) features"""
        print("📺 Creating channel features...")
        
        # Clicks per channel
        channel_counts = df.groupby('channel').size().reset_index(name='channel_click_count')
        df = df.merge(channel_counts, on='channel', how='left')
        
        # Channel conversion rate
        channel_conversion = df.groupby('channel')['is_attributed'].agg(['sum', 'count']).reset_index()
        channel_conversion['channel_conversion_rate'] = channel_conversion['sum'] / channel_conversion['count']
        df = df.merge(channel_conversion[['channel', 'channel_conversion_rate']], on='channel', how='left')
        
        return df
    
    def create_interaction_features(self, df):
        """Combination features"""
        print("🔗 Creating interaction features...")
        
        # IP + App combination count
        ip_app_counts = df.groupby(['ip', 'app']).size().reset_index(name='ip_app_combo_count')
        df = df.merge(ip_app_counts, on=['ip', 'app'], how='left')
        
        # IP + Device + OS combination
        ip_device_os = df.groupby(['ip', 'device', 'os']).size().reset_index(name='ip_device_os_count')
        df = df.merge(ip_device_os, on=['ip', 'device', 'os'], how='left')
        
        return df
    
    def create_all_features(self, df):
        """Create all fraud detection features"""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*50)
        
        initial_cols = len(df.columns)
        
        df = self.create_time_features(df)
        df = self.create_ip_features(df)
        df = self.create_app_features(df)
        df = self.create_channel_features(df)
        df = self.create_interaction_features(df)
        
        final_cols = len(df.columns)
        print(f"\n✅ Created {final_cols - initial_cols} new features")
        print(f"Total features: {final_cols}")
        print("="*50 + "\n")
        
        return df
    
    def save_processed_data(self, df, output_path):
        """Save processed data"""
        df.to_csv(output_path, index=False)
        print(f"💾 Saved processed data to {output_path}")

if __name__ == "__main__":
    # Import here to avoid circular imports
    from src.ingestion.data_loader import DataLoader
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    df = loader.basic_cleaning(df)
    
    # Create features
    engineer = FraudFeatureEngine()
    df = engineer.create_all_features(df)
    
    # Save
    output_path = os.getenv('PROCESSED_DATA_PATH')
    engineer.save_processed_data(df, output_path)
    
    print("\n📊 Sample of processed data:")
    print(df[['ip', 'app', 'ip_click_count', 'ip_app_count', 'is_attributed']].head())
