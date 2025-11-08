import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

load_dotenv()

class FraudModelTrainer:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.smote = SMOTE(random_state=42, sampling_strategy=0.3)
        
    def prepare_features(self, df):
        """Select and prepare features for training"""
        print("🔧 Preparing features...")
        
        # Drop non-feature columns
        drop_cols = ['click_time', 'is_attributed']
        
        # Select feature columns (all numeric columns except target)
        feature_cols = [col for col in df.columns if col not in drop_cols]
        
        X = df[feature_cols]
        y = df['is_attributed']
        
        self.feature_columns = feature_cols
        
        print(f"✅ Features prepared: {len(feature_cols)} features")
        print(f"   Feature list: {', '.join(feature_cols[:10])}...")
        
        return X, y
    
    def handle_imbalance(self, X, y):
        """Apply SMOTE to handle class imbalance"""
        print("\n⚖️ Handling class imbalance...")
        print(f"   Before SMOTE - Fraud: {y.sum()}, Legitimate: {(y==0).sum()}")
        print(f"   Fraud rate: {y.mean()*100:.3f}%")
        
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        print(f"   After SMOTE - Fraud: {y_resampled.sum()}, Legitimate: {(y_resampled==0).sum()}")
        print(f"   Fraud rate: {y_resampled.mean()*100:.3f}%")
        
        return X_resampled, y_resampled
    
    def train_model(self, X_train, y_train):
        """Train the fraud detection model"""
        print(f"\n🎯 Training {self.model_type} model...")
        
        if self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=10,  # Handle imbalance
                random_state=42,
                n_jobs=-1
            )
        else:  # random_forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_train, y_train)
        print("✅ Model training complete!")
        
        return self.model
    
    def save_model(self, model_path=None):
        """Save trained model to disk"""
        if model_path is None:
            model_path = os.getenv('MODEL_PATH', 'data/models/fraud_detector.pkl')
        
        # Create directory if doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and feature columns
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved to: {model_path}")
    
    def get_feature_importance(self, top_n=15):
        """Get top important features"""
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_imp.head(top_n)
        return None

def main():
    print("="*60)
    print("FRAUD DETECTION MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load processed data
    print("\n1️⃣ Loading processed data...")
    processed_path = os.getenv('PROCESSED_DATA_PATH')
    
    if not os.path.exists(processed_path):
        print(f"❌ Processed data not found at: {processed_path}")
        print("   Run feature engineering first: python run_feature_engineering.py")
        return
    
    df = pd.read_csv(processed_path)
    print(f"✅ Loaded {len(df):,} rows")
    
    # Initialize trainer
    trainer = FraudModelTrainer(model_type='xgboost')
    
    # Prepare features
    print("\n2️⃣ Preparing features...")
    X, y = trainer.prepare_features(df)
    
    # Split data
    print("\n3️⃣ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train):,} rows")
    print(f"   Test set: {len(X_test):,} rows")
    
    # Handle imbalance
    print("\n4️⃣ Handling class imbalance...")
    X_train_balanced, y_train_balanced = trainer.handle_imbalance(X_train, y_train)
    
    # Train model
    print("\n5️⃣ Training model...")
    model = trainer.train_model(X_train_balanced, y_train_balanced)
    
    # Save model
    print("\n6️⃣ Saving model...")
    trainer.save_model()
    
    # Feature importance
    print("\n7️⃣ Top 15 Important Features:")
    feature_imp = trainer.get_feature_importance()
    if feature_imp is not None:
        print(feature_imp.to_string(index=False))
    
    # Save test data for evaluation
    print("\n8️⃣ Saving test data for evaluation...")
    test_data = pd.DataFrame(X_test, columns=trainer.feature_columns)
    test_data['is_attributed'] = y_test.values
    test_data.to_csv('data/processed/test_data.csv', index=False)
    print("✅ Test data saved to: data/processed/test_data.csv")
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  → Run evaluation: python src/models/evaluate.py")
    print("  → Or use wrapper: python run_model_evaluation.py")

if __name__ == "__main__":
    main()
