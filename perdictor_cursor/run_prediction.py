#!/usr/bin/env python3
"""
Simple script to run stock price predictions
"""

from stock_predictor import StockPredictor
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_prediction.py <SYMBOL> [MINUTES]")
        print("Example: python run_prediction.py NSE:RELIANCE-EQ 15")
        print("\nAvailable symbols (top 10 by volume):")
        print("NSE:NIFTY50-INDEX, NSE:TATASTEEL-EQ, NSE:INFY-EQ, NSE:HDFCBANK-EQ")
        print("NSE:ICICIBANK-EQ, NSE:RELIANCE-EQ, NSE:TATAMOTORS-EQ, NSE:BAJFINANCE-EQ")
        print("NSE:M&M-EQ, NSE:TCS-EQ")
        return 1
    
    symbol = sys.argv[1]
    minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    
    print(f"Initializing Stock Predictor for {symbol}...")
    predictor = StockPredictor()
    
    try:
        # Try to load symbol-specific models first
        if predictor.load_models(symbol):
            print(f"Loaded existing trained models for {symbol}")
            result = predictor.predict(symbol, minutes)
        else:
            print(f"No existing models found for {symbol}. Training new models...")
            result = predictor.train_and_predict(symbol, minutes)
        
        # Display results
        print("\n" + "="*70)
        print("📈 STOCK PRICE PREDICTION RESULTS")
        print("="*70)
        print(f"🏷️  Symbol: {result['symbol']}")
        print(f"💰 Current Price: ₹{result['current_price']:.2f}")
        print(f"🔮 Predicted Price (Next {minutes} mins): ₹{result['predicted_price']:.2f}")
        
        change = result['price_change']
        change_pct = result['price_change_pct']
        if change > 0:
            print(f"📈 Expected Change: +₹{change:.2f} (+{change_pct:.2f}%)")
        else:
            print(f"📉 Expected Change: ₹{change:.2f} ({change_pct:.2f}%)")
        
        print(f"⏰ Last Trade Time: {result['last_trade_time']}")
        print(f"🕐 Prediction Time: {result['prediction_time']}")
        print(f"🎯 Confidence: {result['confidence']:.1%}")
        
        print("\n🤖 Individual Model Predictions:")
        for model, pred in result['individual_predictions'].items():
            if pred is not None:
                print(f"   {model.replace('_', ' ').title()}: ₹{pred:.2f}")
        
        print("="*70)
        
        # Additional insights
        if result['confidence'] > 0.7:
            print("✅ High confidence prediction")
        elif result['confidence'] > 0.5:
            print("⚠️  Medium confidence prediction")
        else:
            print("❌ Low confidence prediction - consider more data")
            
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
