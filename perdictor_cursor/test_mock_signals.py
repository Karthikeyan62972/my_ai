#!/usr/bin/env python3
"""
Test script to create mock data and test trading signal generation
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

def create_strong_mock_data():
    """Create mock data with strong price movements to trigger signals"""
    
    # Connect to market database
    conn = sqlite3.connect('/home/karthik/market.db')
    
    # Test with a specific symbol that we know works
    test_symbol = 'NSE:SBIN-EQ'
    
    print(f"Creating strong mock data for {test_symbol}...")
    
    # Get latest data
    latest_df = pd.read_sql_query(
        "SELECT ts, ltp FROM ticks WHERE symbol = ? ORDER BY ts DESC LIMIT 1", 
        conn, params=(test_symbol,)
    )
    
    if len(latest_df) > 0:
        latest_ts = pd.to_datetime(latest_df['ts'].iloc[0])
        base_price = latest_df['ltp'].iloc[0]
    else:
        latest_ts = datetime.now()
        base_price = 800.0
    
    print(f"Base price: {base_price}, Latest timestamp: {latest_ts}")
    
    # Create mock data with strong upward trend (should trigger BUY signal)
    mock_data = []
    current_time = latest_ts + timedelta(seconds=1)
    current_price = base_price
    
    # Create a strong upward trend
    for i in range(200):  # Create 200 mock data points
        if i < 50:
            # Initial small movements
            price_change_pct = random.uniform(-0.2, 0.2)
        elif i < 100:
            # Start building momentum
            price_change_pct = random.uniform(0.1, 0.5)
        elif i < 150:
            # Strong upward movement
            price_change_pct = random.uniform(0.3, 1.0)
        else:
            # Continue strong trend
            price_change_pct = random.uniform(0.2, 0.8)
        
        price_change = current_price * (price_change_pct / 100)
        current_price += price_change
        
        # Create mock tick data
        mock_tick = {
            'symbol': test_symbol,
            'ts': current_time.isoformat(),
            'ltp': round(current_price, 2),
            'bid': round(current_price - random.uniform(0.1, 0.5), 2),
            'ask': round(current_price + random.uniform(0.1, 0.5), 2),
            'bid_qty': random.randint(100, 1000),
            'ask_qty': random.randint(100, 1000),
            'vwap': round(current_price * random.uniform(0.99, 1.01), 2),
            'vol_traded_today': random.randint(10000, 100000),
            'last_traded_qty': random.randint(1, 100),
            'exch_feed_time': current_time.isoformat(),
            'bid_size': random.randint(1, 10),
            'ask_size': random.randint(1, 10),
            'bid_price': round(current_price - random.uniform(0.1, 0.5), 2),
            'ask_price': round(current_price + random.uniform(0.1, 0.5), 2),
            'tot_buy_qty': random.randint(1000, 10000),
            'tot_sell_qty': random.randint(1000, 10000),
            'avg_trade_price': round(current_price * random.uniform(0.99, 1.01), 2),
            'ch': round(price_change, 2),
            'chp': round((price_change / (current_price - price_change)) * 100, 2),
            'last_traded_time': current_time.isoformat(),
            'high': round(current_price * random.uniform(1.0, 1.02), 2),
            'low': round(current_price * random.uniform(0.98, 1.0), 2),
            'open': round(current_price * random.uniform(0.99, 1.01), 2),
            'prev_close': round((current_price - price_change) * random.uniform(0.99, 1.01), 2),
            'high_price': round(current_price * random.uniform(1.0, 1.02), 2),
            'low_price': round(current_price * random.uniform(0.98, 1.0), 2),
            'open_price': round(current_price * random.uniform(0.99, 1.01), 2),
            'prev_close_price': round((current_price - price_change) * random.uniform(0.99, 1.01), 2)
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

def test_single_symbol(symbol):
    """Test prediction for a single symbol"""
    from multi_symbol_monitor_v2 import MultiSymbolMonitor
    
    print(f"\nTesting {symbol} with fresh data...")
    
    # Initialize monitor
    monitor = MultiSymbolMonitor(
        db_path='/home/karthik/market.db',
        symbols_file='/home/karthik/new/symbols.txt'
    )
    
    try:
        # Make prediction
        result = monitor.predictor.predict(symbol, 15)
        signal = monitor.predictor.generate_trading_signal(result)
        
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
        return False

def run_monitoring_cycle():
    """Run a full monitoring cycle to see if triggers are created"""
    from multi_symbol_monitor_v2 import MultiSymbolMonitor
    
    print("\n" + "="*60)
    print("RUNNING FULL MONITORING CYCLE")
    print("="*60)
    
    # Initialize monitor
    monitor = MultiSymbolMonitor(
        db_path='/home/karthik/market.db',
        symbols_file='/home/karthik/new/symbols.txt'
    )
    
    # Run monitoring cycle
    monitor._run_cycle()
    
    # Check if any triggers were created
    import sqlite3
    conn = sqlite3.connect('/home/karthik/monitoring.db')
    cursor = conn.execute("SELECT COUNT(*) FROM trading_triggers WHERE status = 'monitoring'")
    trigger_count = cursor.fetchone()[0]
    conn.close()
    
    print(f"Active triggers after cycle: {trigger_count}")

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
    print("Mock Data Signal Testing Script")
    print("=" * 40)
    
    try:
        # Create strong mock data
        test_symbol = create_strong_mock_data()
        
        # Wait a moment for data to be available
        time.sleep(2)
        
        # Test the symbol
        triggered = test_single_symbol(test_symbol)
        
        if triggered:
            print(f"\n🎯 SUCCESS! Trading signal triggered for {test_symbol}")
        else:
            print(f"\n📊 No trading signal triggered for {test_symbol}")
        
        # Run full monitoring cycle
        run_monitoring_cycle()
        
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

