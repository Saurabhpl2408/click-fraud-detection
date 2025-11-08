#!/bin/bash

echo "=================================="
echo "CLICK FRAUD DETECTION PIPELINE"
echo "=================================="

# Activate virtual environment
source venv/bin/activate

# Step 1: Feature Engineering
echo -e "\n[1/3] 🔧 Feature Engineering..."
python run_feature_engineering.py
if [ $? -ne 0 ]; then
    echo "❌ Feature engineering failed!"
    exit 1
fi

# Step 2: Model Training
echo -e "\n[2/3] 🎯 Model Training..."
python run_model_training.py
if [ $? -ne 0 ]; then
    echo "❌ Model training failed!"
    exit 1
fi

# Step 3: Model Evaluation
echo -e "\n[3/3] 📊 Model Evaluation..."
python run_model_evaluation.py
if [ $? -ne 0 ]; then
    echo "❌ Model evaluation failed!"
    exit 1
fi

echo -e "\n=================================="
echo "✅ PIPELINE COMPLETE!"
echo "=================================="
echo ""
echo "📁 Check outputs:"
echo "   - Processed data: data/processed/"
echo "   - Models: data/models/"
echo "   - Visualizations: data/models/*.png"
echo ""
