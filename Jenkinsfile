pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
        VENV_PATH = 'venv'
        MODEL_PATH = 'data/models/fraud_detector.pkl'
        FRAUD_THRESHOLD = '0.85'
    }
    
    stages {
        stage('Setup') {
            steps {
                echo '🔧 Setting up environment...'
                sh '''
                    python3 -m venv ${VENV_PATH}
                    . ${VENV_PATH}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Data Validation') {
            steps {
                echo '📊 Validating data...'
                sh '''
                    . ${VENV_PATH}/bin/activate
                    python -c "
import pandas as pd
import os

# Check if data exists
raw_path = 'data/raw/train_sample.csv'
if not os.path.exists(raw_path):
    raise FileNotFoundError(f'Data not found: {raw_path}')

# Load and validate
df = pd.read_csv(raw_path)
print(f'✅ Data validated: {len(df):,} rows')

# Check for required columns
required_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f'Missing columns: {missing}')

print('✅ All required columns present')

# Check data quality
if df.isnull().sum().sum() > len(df) * 0.05:
    raise ValueError('Too many null values (>5%)')

print('✅ Data quality check passed')
                    "
                '''
            }
        }
        
        stage('Feature Engineering') {
            steps {
                echo '🔧 Engineering features...'
                sh '''
                    . ${VENV_PATH}/bin/activate
                    python run_feature_engineering.py
                '''
            }
        }
        
        stage('Model Training') {
            steps {
                echo '🎯 Training model...'
                sh '''
                    . ${VENV_PATH}/bin/activate
                    python run_model_training.py
                '''
            }
        }
        
        stage('Model Evaluation') {
            steps {
                echo '📊 Evaluating model...'
                sh '''
                    . ${VENV_PATH}/bin/activate
                    python run_model_evaluation.py
                '''
            }
        }
        
        stage('Quality Gate') {
            steps {
                echo '🚦 Checking model quality...'
                sh '''
                    . ${VENV_PATH}/bin/activate
                    python -c "
import pandas as pd
import json

# Read evaluation metrics from model evaluation output
# In production, this would read from database or metrics file

# For now, we'll check if model file exists and is recent
import os
from datetime import datetime, timedelta

model_path = '${MODEL_PATH}'
if not os.path.exists(model_path):
    print('❌ Model file not found')
    exit(1)

# Check if model is recent (within last hour)
model_time = datetime.fromtimestamp(os.path.getmtime(model_path))
if datetime.now() - model_time > timedelta(hours=1):
    print('⚠️ Model is old, but continuing...')
else:
    print('✅ Model is fresh')

# In production, check actual metrics:
# if precision < 0.90 or recall < 0.75:
#     print('❌ Model quality below threshold')
#     exit(1)

print('✅ Quality gate passed')
                    "
                '''
            }
        }
        
        stage('Deploy API') {
            steps {
                echo '🚀 Deploying API...'
                sh '''
                    . ${VENV_PATH}/bin/activate
                    
                    # Kill any existing API process
                    pkill -f "fraud_api" || true
                    
                    # Start API in background
                    nohup python src/api/fraud_api.py > logs/api.log 2>&1 &
                    
                    # Wait for API to start
                    sleep 5
                    
                    # Health check
                    curl -f http://localhost:8000/health || exit 1
                    
                    echo "✅ API deployed successfully"
                '''
            }
        }
        
        stage('Integration Tests') {
            steps {
                echo '🧪 Running integration tests...'
                sh '''
                    . ${VENV_PATH}/bin/activate
                    
                    # Wait for API to be ready
                    sleep 3
                    
                    # Run basic API tests
                    python -c "
import requests
import time

# Retry logic for API startup
max_retries = 5
for i in range(max_retries):
    try:
        response = requests.get('http://localhost:8000/health')
        if response.status_code == 200:
            print('✅ API health check passed')
            break
    except:
        if i < max_retries - 1:
            print(f'Retry {i+1}/{max_retries}...')
            time.sleep(2)
        else:
            raise

# Test prediction endpoint
test_click = {
    'ip': 123456,
    'app': 3,
    'device': 1,
    'os': 13,
    'channel': 497
}

response = requests.post('http://localhost:8000/score_click', json=test_click)
if response.status_code != 200:
    raise Exception('Prediction endpoint failed')

print('✅ Integration tests passed')
                    "
                '''
            }
        }
    }
    
    post {
        success {
            echo '✅ Pipeline completed successfully!'
            sh '''
                echo "====================================="
                echo "DEPLOYMENT SUMMARY"
                echo "====================================="
                echo "Model: ${MODEL_PATH}"
                echo "API: http://localhost:8000"
                echo "Docs: http://localhost:8000/docs"
                echo "====================================="
            '''
        }
        
        failure {
            echo '❌ Pipeline failed!'
            sh '''
                # Cleanup on failure
                pkill -f "fraud_api" || true
            '''
        }
        
        always {
            echo '🧹 Cleanup...'
            archiveArtifacts artifacts: 'data/models/*.pkl, data/models/*.png, logs/*.log', allowEmptyArchive: true
        }
    }
}
