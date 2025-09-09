#!/usr/bin/env python3
"""
Demo script showing the Stock Price Predictor capabilities
"""

from stock_predictor import StockPredictor
import time

def demo_predictions():
    """Demonstrate predictions for multiple symbols"""
    
    # Popular symbols to test
    symbols = [
        'NSE:RELIANCE-EQ',
        'NSE:INFY-EQ', 
        'NSE:HDFCBANK-EQ',
        'NSE:TCS-EQ',
        'NSE:ICICIBANK-EQ'
    ]
    
    print("🚀 Stock Price Predictor Demo")
    print("=" * 50)
    print("Testing predictions for multiple symbols...")
    print()
    
    predictor = StockPredictor()
    
    # Note: We'll train models for each symbol individually for better accuracy
    
    results = []
    
    for symbol in symbols:
        try:
            print(f"📊 Predicting for {symbol}...")
            
            # Try to load symbol-specific models, train if not available
            if not predictor.load_models(symbol):
                print(f"   Training models for {symbol}...")
                result = predictor.train_and_predict(symbol, 15)
            else:
                result = predictor.predict(symbol, 15)
            
            results.append(result)
            
            # Display result
            change_emoji = "📈" if result['price_change'] > 0 else "📉"
            print(f"   {change_emoji} Current: ₹{result['current_price']:.2f} → Predicted: ₹{result['predicted_price']:.2f}")
            print(f"   💡 Change: {result['price_change']:+.2f} ({result['price_change_pct']:+.2f}%)")
            print(f"   🎯 Confidence: {result['confidence']:.1%}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
    
    # Summary
    if results:
        print("📋 SUMMARY")
        print("=" * 50)
        
        bullish = sum(1 for r in results if r['price_change'] > 0)
        bearish = len(results) - bullish
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        print(f"📈 Bullish predictions: {bullish}")
        print(f"📉 Bearish predictions: {bearish}")
        print(f"🎯 Average confidence: {avg_confidence:.1%}")
        
        # Best and worst predictions
        best = max(results, key=lambda x: x['price_change'])
        worst = min(results, key=lambda x: x['price_change'])
        
        print(f"🏆 Best prediction: {best['symbol']} (+{best['price_change']:.2f})")
        print(f"📉 Worst prediction: {worst['symbol']} ({worst['price_change']:.2f})")
        
        print("\n⚠️  DISCLAIMER: These are predictions for educational purposes only.")
        print("   Do not use for actual trading decisions without proper analysis.")

if __name__ == "__main__":
    demo_predictions()
