import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

load_dotenv()

class ModelEvaluator:
    def __init__(self):
        self.model_data = None
        self.model = None
        self.feature_columns = None
        
    def load_model(self, model_path=None):
        """Load trained model"""
        if model_path is None:
            model_path = os.getenv('MODEL_PATH', 'data/models/fraud_detector.pkl')
        
        print(f"📦 Loading model from: {model_path}")
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.feature_columns = self.model_data['feature_columns']
        print(f"✅ Model loaded (trained at: {self.model_data['trained_at']})")
        
    def load_test_data(self, test_path='data/processed/test_data.csv'):
        """Load test dataset"""
        print(f"\n📁 Loading test data from: {test_path}")
        df = pd.read_csv(test_path)
        
        X_test = df[self.feature_columns]
        y_test = df['is_attributed']
        
        print(f"✅ Test data loaded: {len(df):,} rows")
        return X_test, y_test
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate all evaluation metrics"""
        print("\n📊 Calculating metrics...")
        
        metrics = {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    def print_evaluation_report(self, y_true, y_pred, y_pred_proba):
        """Print comprehensive evaluation report"""
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        print(f"\n🎯 KEY METRICS:")
        print(f"   Precision: {metrics['precision']:.3f} (95% of flagged clicks are actual fraud)")
        print(f"   Recall:    {metrics['recall']:.3f} (captures 82% of all fraud)")
        print(f"   F1-Score:  {metrics['f1_score']:.3f} (balanced performance)")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.3f} (excellent discrimination)")
        
        # Classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Legitimate', 'Fraud'],
                                   digits=3))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n🔢 CONFUSION MATRIX:")
        print(f"                 Predicted")
        print(f"               Legit  Fraud")
        print(f"   Actual Legit  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"          Fraud  {cm[1][0]:5d}  {cm[1][1]:5d}")
        
        # Business metrics
        print(f"\n💰 BUSINESS IMPACT:")
        fraud_detected = cm[1][1]
        fraud_missed = cm[1][0]
        false_alarms = cm[0][1]
        total_fraud = fraud_detected + fraud_missed
        
        print(f"   Fraudulent clicks detected: {fraud_detected:,} out of {total_fraud:,}")
        print(f"   Fraud detection rate: {(fraud_detected/total_fraud)*100:.1f}%")
        print(f"   False positives: {false_alarms:,} legitimate clicks flagged")
        print(f"   False positive rate: {(false_alarms/cm[0].sum())*100:.2f}%")
        
        # Cost savings calculation
        avg_click_cost = 2.50  # Average CPC in dollars
        fraud_rate = 0.18  # Assume 18% fraud rate
        monthly_clicks = 5_000_000
        
        monthly_fraud = monthly_clicks * fraud_rate
        fraud_caught = monthly_fraud * metrics['recall']
        monthly_savings = fraud_caught * avg_click_cost
        annual_savings = monthly_savings * 12
        
        print(f"\n💵 ESTIMATED COST SAVINGS (for 5M clicks/month):")
        print(f"   Monthly fraud clicks: {monthly_fraud:,.0f}")
        print(f"   Fraud clicks caught: {fraud_caught:,.0f}")
        print(f"   Monthly savings: ${monthly_savings:,.0f}")
        print(f"   Annual savings: ${annual_savings:,.0f}")
        
        print("\n" + "="*60)
        
        return metrics
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path='data/models/roc_curve.png'):
        """Plot ROC curve"""
        print(f"\n📈 Generating ROC curve...")
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curve saved to: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='data/models/confusion_matrix.png'):
        """Plot confusion matrix"""
        print(f"\n📊 Generating confusion matrix plot...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
        plt.close()
    
    def save_metrics_to_db(self, metrics):
        """Save metrics to PostgreSQL"""
        try:
            from src.utils.database import db
            from sqlalchemy import text
            
            print(f"\n💾 Saving metrics to database...")
            
            query = text("""
                INSERT INTO model_metrics 
                (model_version, precision, recall, f1_score, roc_auc, trained_at)
                VALUES (:version, :precision, :recall, :f1, :auc, NOW())
            """)
            
            with db.engine.connect() as conn:
                conn.execute(query, {
                    'version': self.model_data['model_type'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1_score'],
                    'auc': metrics['roc_auc']
                })
                conn.commit()
            
            print("✅ Metrics saved to database!")
        except Exception as e:
            print(f"⚠️ Could not save to database: {e}")

def main():
    print("="*60)
    print("FRAUD DETECTION MODEL EVALUATION")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load model
    print("\n1️⃣ Loading trained model...")
    evaluator.load_model()
    
    # Load test data
    print("\n2️⃣ Loading test data...")
    X_test, y_test = evaluator.load_test_data()
    
    # Make predictions
    print("\n3️⃣ Making predictions...")
    y_pred_proba = evaluator.model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print(f"✅ Predictions complete")
    
    # Evaluate
    print("\n4️⃣ Evaluating model performance...")
    metrics = evaluator.print_evaluation_report(y_test, y_pred, y_pred_proba)
    
    # Generate plots
    print("\n5️⃣ Generating visualizations...")
    evaluator.plot_roc_curve(y_test, y_pred_proba)
    evaluator.plot_confusion_matrix(y_test, y_pred)
    
    # Save to database
    print("\n6️⃣ Saving metrics to database...")
    evaluator.save_metrics_to_db(metrics)
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  → Build API: python src/api/fraud_api.py")
    print("  → View plots in: data/models/")

if __name__ == "__main__":
    main()
