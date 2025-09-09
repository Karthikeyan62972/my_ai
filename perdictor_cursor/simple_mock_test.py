#!/usr/bin/env python3
"""
Simple test to verify trading signal generation with mock data
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

def create_simple_mock_data():
    """Create simple mock data with clear price movements"""
    
    # Connect to market database
    conn = sqlite3.connect('/home/karthik/market.db')
    
    # Test with SBIN
    test_symbol = 'NSE:SBIN-EQ'
    
    print(f"Creating simple mock data for {test_symbol}...")
    
    # Get latest data
    latest_df = pd.read_sql_query(
        "SELECT ts, ltp FROM ticks WHERE symbol = ? ORDER BY ts DESC LIMIT 1", 
        conn, params=(test_symbol,)
    )
    
    if len(latest_df) > 0:
        latest_ts = pd.to_datetime(latest_df['ts'].iloc[0])
        base_price = float(latest_df['ltp'].iloc[0])
    else:
        latest_ts = datetime.now()
        base_price = 800.0
    
    print(f"Base price: {base_price}, Latest timestamp: {latest_ts}")
    
    # Create simple mock data with clear upward trend
    mock_data = []
    current_time = latest_ts + timedelta(seconds=1)
    current_price = base_price
    
    # Create a clear upward trend
    for i in range(50):  # Create 50 mock data points
        # Simple upward trend: 0.5% increase per step
        price_change_pct = 0.5  # 0.5% increase
        price_change = current_price * (price_change_pct / 100)
        current_price += price_change
        
        # Create mock tick data with proper data types
        mock_tick = {
            'symbol': test_symbol,
            'ts': current_time.isoformat(),
            'ltp': float(round(current_price, 2)),
            'bid': float(round(current_price - 0.5, 2)),
            'ask': float(round(current_price + 0.5, 2)),
            'bid_qty': int(random.randint(100, 1000)),
            'ask_qty': int(random.randint(100, 1000)),
            'vwap': float(round(current_price, 2)),
            'vol_traded_today': int(random.randint(10000, 100000)),
            'last_traded_qty': int(random.randint(1, 100)),
            'exch_feed_time': current_time.isoformat(),
            'bid_size': int(random.randint(1, 10)),
            'ask_size': int(random.randint(1, 10)),
            'bid_price': float(round(current_price - 0.5, 2)),
            'ask_price': float(round(current_price + 0.5, 2)),
            'tot_buy_qty': int(random.randint(1000, 10000)),
            'tot_sell_qty': int(random.randint(1000, 10000)),
            'avg_trade_price': float(round(current_price, 2)),
            'ch': float(round(price_change, 2)),
            'chp': float(round((price_change / (current_price - price_change)) * 100, 2)),
            'last_traded_time': current_time.isoformat(),
            'high': float(round(current_price * 1.01, 2)),
            'low': float(round(current_price * 0.99, 2)),
            'open': float(round(current_price, 2)),
            'prev_close': float(round(current_price - price_change, 2)),
            'high_price': float(round(current_price * 1.01, 2)),
            'low_price': float(round(current_price * 0.99, 2)),
            'open_price': float(round(current_price, 2)),
            'prev_close_price': float(round(current_price - price_change, 2))
        }
        
        mock_data.append(mock_tick)
        current_time += timedelta(seconds=1)
    
    # Insert mock data into database
    df = pd.DataFrame(mock_data)
    df.to_sql('ticks', conn, if_exists='append', index=False)
    
    conn.close()
    print(f"Inserted {len(mock_data)} mock records for {test_symbol}")
    print(f"Price range: {base_price:.2f} → {current_price:.2f} ({((current_price - base_price) / base_price * 100):.2f}% change)")
    
    return test_symbol

def test_prediction():
    """Test prediction with the mock data"""
    from optimized_predictor_v2 import RobustStockPredictor
    
    test_symbol = 'NSE:SBIN-EQ'
    print(f"\nTesting prediction for {test_symbol}...")
    
    # Initialize predictor
    predictor = RobustStockPredictor('/home/karthik/market.db')
    
    try:
        # Load models
        if not predictor.load_models(test_symbol):
            print("Training new models...")
            # Get some data for training
            df = predictor.fetch_data(test_symbol, 1000)
            predictor.train_models(test_symbol, df, 15)
        
        # Make prediction
        result = predictor.predict(test_symbol, 15)
        signal = predictor.generate_trading_signal(result)
        
        print(f"  Current Price: {result['current_price']:.2f}")
        print(f"  Predicted Price: {result['predicted_price']:.2f}")
        print(f"  Price Change: {result['price_change_pct']:.2f}%")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Signal: {signal['signal']}")
        print(f"  Potential Profit: {signal['potential_profit_pct']:.2f}%")
        
        # Check if this would trigger a trading signal
        if (signal['confidence'] >= 0.8 and 
            signal['potential_profit_pct'] >= 0.5):
            print(f"  🎯 WOULD TRIGGER TRADING SIGNAL!")
            return True
        else:
            print(f"  📊 Regular monitoring (no trigger)")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_monitoring_test():
    """Run the monitoring system to see if it creates triggers"""
    print("\n" + "="*60)
    print("RUNNING MONITORING SYSTEM TEST")
    print("="*60)
    
    # Run the monitoring system once
    import subprocess
    result = subprocess.run(['python3', 'multi_symbol_monitor_v2.py', '--once'], 
                          capture_output=True, text=True)
    
    print("Monitoring output:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    # Check if any triggers were created
    conn = sqlite3.connect('/home/karthik/monitoring.db')
    cursor = conn.execute("SELECT COUNT(*) FROM trading_triggers WHERE status = 'monitoring'")
    trigger_count = cursor.fetchone()[0]
    conn.close()
    
    print(f"\nActive triggers after monitoring: {trigger_count}")

def cleanup_mock_data():
    """Remove mock data from database"""
    conn = sqlite3.connect('/home/karthik/market.db')
    
    # Get mock data timestamps (recent ones)
    cutoff_time = (datetime.now() - timedelta(minutes=5)).isoformat()
    
    # Delete mock data
    cursor = conn.execute("DELETE FROM ticks WHERE ts > ?", (cutoff_time,))
    deleted_count = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    print(f"Cleaned up {deleted_count} mock data records")

if __name__ == "__main__":
    print("Simple Mock Data Test")
    print("=" * 30)
    
    try:
        # Create mock data
        test_symbol = create_simple_mock_data()
        
        # Wait a moment for data to be available
        time.sleep(2)
        
        # Test prediction
        triggered = test_prediction()
        
        if triggered:
            print(f"\n🎯 SUCCESS! Trading signal triggered for {test_symbol}")
        else:
            print(f"\n📊 No trading signal triggered for {test_symbol}")
        
        # Run monitoring test
        run_monitoring_test()
        
        # Ask user if they want to clean up
        print("\n" + "="*60)
        cleanup_choice = input("Do you want to clean up the mock data? (y/n): ").lower()
        if cleanup_choice == 'y':
            cleanup_mock_data()
            print("Mock data cleaned up!")
        else:
            print("Mock data left in database for further testing")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
