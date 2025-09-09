#!/bin/bash

echo "Installing Stock Price Predictor Dependencies"
echo "=============================================="

# Core dependencies
echo "Installing core dependencies..."
pip3 install numpy pandas scikit-learn joblib

# Optional but recommended dependencies
echo "Installing optional dependencies for better performance..."
pip3 install xgboost lightgbm ta matplotlib seaborn

echo ""
echo "Installation complete!"
echo ""
echo "To test the installation, run:"
echo "python3 run_prediction.py NSE:RELIANCE-EQ 15"
echo ""
echo "For more information, see README.md"

