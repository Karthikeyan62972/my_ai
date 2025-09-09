#!/usr/bin/env python3
"""
Test script with mock data to verify trading signal generation
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

def create_mock_data():
    """Create mock market data that should trigger trading signals"""
    
    # Connect to market database
    conn = sqlite3.connect('/home/karthik/market.db')
    
    # Get existing symbols
    symbols_df = pd.read_sql_query("SELECT DISTINCT symbol FROM ticks LIMIT 5", conn)
    symbols = symbols_df['symbol'].tolist()
    
    print(f"Creating mock data for symbols: {symbols}")
    
    # Create mock data for each symbol
    for symbol in symbols:
        print(f"Creating mock data for {symbol}...")
        
        # Get latest timestamp from existing data
        latest_df = pd.read_sql_query(
            "SELECT ts FROM ticks WHERE symbol = ? ORDER BY ts DESC LIMIT 1", 
            conn, params=(symbol,)
        )
        
        if len(latest_df) > 0:
            latest_ts = pd.to_datetime(latest_df['ts'].iloc[0])
        else:
            latest_ts = datetime.now()
        
        # Get latest price
        latest_price_df = pd.read_sql_query(
            "SELECT ltp FROM ticks WHERE symbol = ? ORDER BY ts DESC LIMIT 1", 
            conn, params=(symbol,)
        )
        
        if len(latest_price_df) > 0:
            base_price = latest_price_df['ltp'].iloc[0]
        else:
            base_price = 1000.0
        
        # Create mock data points with significant price movements
        mock_data = []
        current_time = latest_ts + timedelta(seconds=1)
        current_price = base_price
        
        for i in range(100):  # Create 100 mock data points
            # Create significant price movements (2-5% changes)
            if i % 20 == 0:  # Every 20th point, create a big movement
                price_change_pct = random.uniform(2.0, 5.0)  # 2-5% change
                direction = random.choice([-1, 1])  # Random direction
                price_change = current_price * (price_change_pct / 100) * direction
            else:
                # Small random movements
                price_change_pct = random.uniform(-0.5, 0.5)  # -0.5% to 0.5%
                price_change = current_price * (price_change_pct / 100)
            
            current_price += price_change
            current_price = max(current_price, base_price * 0.5)  # Don't go below 50% of base
            
            # Create mock tick data
            mock_tick = {
                'symbol': symbol,
                'ts': current_time.isoformat(),
                'ltp': round(current_price, 2),
                'bid': round(current_price - random.uniform(0.1, 1.0), 2),
                'ask': round(current_price + random.uniform(0.1, 1.0), 2),
                'bid_qty': random.randint(100, 1000),
                'ask_qty': random.randint(100, 1000),
                'vwap': round(current_price * random.uniform(0.98, 1.02), 2),
                'vol_traded_today': random.randint(10000, 100000),
                'last_traded_qty': random.randint(1, 100),
                'exch_feed_time': current_time.isoformat(),
                'bid_size': random.randint(1, 10),
                'ask_size': random.randint(1, 10),
                'bid_price': round(current_price - random.uniform(0.1, 1.0), 2),
                'ask_price': round(current_price + random.uniform(0.1, 1.0), 2),
                'tot_buy_qty': random.randint(1000, 10000),
                'tot_sell_qty': random.randint(1000, 10000),
                'avg_trade_price': round(current_price * random.uniform(0.99, 1.01), 2),
                'ch': round(price_change, 2),
                'chp': round((price_change / (current_price - price_change)) * 100, 2),
                'last_traded_time': current_time.isoformat(),
                'high': round(current_price * random.uniform(1.0, 1.05), 2),
                'low': round(current_price * random.uniform(0.95, 1.0), 2),
                'open': round(current_price * random.uniform(0.98, 1.02), 2),
                'prev_close': round((current_price - price_change) * random.uniform(0.98, 1.02), 2),
                'high_price': round(current_price * random.uniform(1.0, 1.05), 2),
                'low_price': round(current_price * random.uniform(0.95, 1.0), 2),
                'open_price': round(current_price * random.uniform(0.98, 1.02), 2),
                'prev_close_price': round((current_price - price_change) * random.uniform(0.98, 1.02), 2)
            }
            
            mock_data.append(mock_tick)
            current_time += timedelta(seconds=1)
        
        # Insert mock data into database
        df = pd.DataFrame(mock_data)
        df.to_sql('ticks', conn, if_exists='append', index=False)
        print(f"Inserted {len(mock_data)} mock records for {symbol}")
    
    conn.close()
    print("Mock data creation completed!")

def test_predictions_with_mock_data():
    """Test the prediction system with mock data"""
    from multi_symbol_monitor_v2 import MultiSymbolMonitor
    
    print("\n" + "="*60)
    print("TESTING PREDICTIONS WITH MOCK DATA")
    print("="*60)
    
    # Initialize monitor
    monitor = MultiSymbolMonitor(
        db_path='/home/karthik/market.db',
        symbols_file='/home/karthik/new/symbols.txt'
    )
    
    # Test with first 3 symbols
    test_symbols = monitor.symbols[:3]
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")
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
            else:
                print(f"  📊 Regular monitoring (no trigger)")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def cleanup_mock_data():
    """Remove mock data from database"""
    conn = sqlite3.connect('/home/karthik/market.db')
    
    # Get mock data timestamps (recent ones)
    cutoff_time = (datetime.now() - timedelta(minutes=10)).isoformat()
    
    # Delete mock data
    cursor = conn.execute("DELETE FROM ticks WHERE ts > ?", (cutoff_time,))
    deleted_count = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    print(f"Cleaned up {deleted_count} mock data records")

if __name__ == "__main__":
    print("Mock Data Testing Script")
    print("=" * 40)
    
    try:
        # Create mock data
        create_mock_data()
        
        # Wait a moment for data to be available
        time.sleep(2)
        
        # Test predictions
        test_predictions_with_mock_data()
        
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

