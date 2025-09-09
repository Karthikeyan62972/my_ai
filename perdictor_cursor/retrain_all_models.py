#!/usr/bin/env python3
"""
Retrain all models with the updated feature set
"""

import os
import sys
from pathlib import Path
from optimized_predictor import OptimizedStockPredictor

def retrain_all_models():
    """Retrain all models to fix feature mismatch issues"""
    
    # Initialize predictor
    predictor = OptimizedStockPredictor('/home/karthik/market.db')
    
    # Load symbols
    symbols_file = '/home/karthik/new/symbols.txt'
    if not os.path.exists(symbols_file):
        print(f"❌ Symbols file not found: {symbols_file}")
        return
    
    with open(symbols_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    print(f"🔄 Retraining models for {len(symbols)} symbols...")
    
    success_count = 0
    error_count = 0
    
    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"[{i}/{len(symbols)}] Training {symbol}...")
            
            # Force retrain by removing existing models
            symbol_clean = symbol.replace(':', '_').replace('-', '_')
            model_dir = predictor.model_dir / symbol_clean
            
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                print(f"   Removed existing models for {symbol}")
            
            # Train new models
            predictor.train_and_predict(symbol, 15)
            success_count += 1
            print(f"   ✅ Successfully trained {symbol}")
            
        except Exception as e:
            error_count += 1
            print(f"   ❌ Error training {symbol}: {e}")
    
    print(f"\n🎉 Retraining completed!")
    print(f"✅ Successfully trained: {success_count}")
    print(f"❌ Errors: {error_count}")
    
    if error_count == 0:
        print("🎯 All models are now using the correct feature set!")
    else:
        print("⚠️  Some models failed to train. Check the errors above.")

if __name__ == "__main__":
    retrain_all_models()

