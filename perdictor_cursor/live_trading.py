#!/usr/bin/env python3
"""
Live Trading Script - Real-time stock predictions during market hours
"""

from realtime_predictor import RealTimeStockPredictor
import time
import sys
from datetime import datetime

def is_market_hours():
    """Check if it's market hours (9:15 AM to 3:30 PM IST)"""
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    
    # Market hours: 9:15 AM to 3:30 PM IST
    if hour == 9 and minute >= 15:
        return True
    elif 10 <= hour <= 14:
        return True
    elif hour == 15 and minute <= 30:
        return True
    
    return False

def live_trading_loop(symbol, update_interval=30):
    """Main live trading loop"""
    
    print(f"🚀 Starting Live Trading Predictions for {symbol}")
    print("=" * 60)
    
    # Initialize predictor
    predictor = RealTimeStockPredictor()
    
    # Load or train initial models
    if not predictor.load_models(symbol):
        print(f"📚 Training initial models for {symbol}...")
        predictor.train_and_predict(symbol, 15)
        print("✅ Initial training completed!")
    
    # Start background updates
    predictor.start_background_updates()
    
    print(f"🔄 Starting continuous predictions (every {update_interval} seconds)")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        while True:
            current_time = datetime.now()
            
            # Check if market is open
            if not is_market_hours():
                print(f"⏰ Market closed. Next update in 1 hour...")
                time.sleep(3600)  # Wait 1 hour
                continue
            
            try:
                # Make prediction
                result = predictor.predict_fast(symbol, 15)
                
                # Display results
                change_emoji = "📈" if result['price_change'] > 0 else "📉"
                confidence_emoji = "🟢" if result['confidence'] > 0.7 else "🟡" if result['confidence'] > 0.5 else "🔴"
                
                print(f"\n[{current_time.strftime('%H:%M:%S')}] {symbol}")
                print(f"💰 Current: ₹{result['current_price']:.2f}")
                print(f"🔮 Predicted: ₹{result['predicted_price']:.2f}")
                print(f"{change_emoji} Change: {result['price_change']:+.2f} ({result['price_change_pct']:+.2f}%)")
                print(f"{confidence_emoji} Confidence: {result['confidence']:.1%}")
                
                # Trading signals
                if result['price_change_pct'] > 0.5 and result['confidence'] > 0.7:
                    print("🟢 BUY SIGNAL: Strong upward movement predicted")
                elif result['price_change_pct'] < -0.5 and result['confidence'] > 0.7:
                    print("🔴 SELL SIGNAL: Strong downward movement predicted")
                elif abs(result['price_change_pct']) < 0.1:
                    print("⚪ HOLD: Minimal movement expected")
                
                print("-" * 40)
                
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                time.sleep(10)
                continue
            
            # Wait for next update
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping live trading...")
        predictor.stop_background_updates()
        print("✅ Live trading stopped successfully")

def main():
    if len(sys.argv) < 2:
        print("Usage: python live_trading.py <SYMBOL> [UPDATE_INTERVAL]")
        print("Example: python live_trading.py NSE:RELIANCE-EQ 30")
        print("\nUpdate interval: seconds between predictions (default: 30)")
        return 1
    
    symbol = sys.argv[1]
    update_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    # Validate update interval
    if update_interval < 10:
        print("⚠️  Warning: Update interval too fast (< 10 seconds)")
        print("   This may cause high CPU usage and database load")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return 1
    
    live_trading_loop(symbol, update_interval)
    return 0

if __name__ == "__main__":
    exit(main())

