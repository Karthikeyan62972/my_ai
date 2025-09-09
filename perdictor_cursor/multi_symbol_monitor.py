#!/usr/bin/env python3
"""
Multi-Symbol Stock Prediction Monitor
Monitors all symbols from symbols.txt and stores predictions in database
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from optimized_predictor import OptimizedStockPredictor

class MultiSymbolMonitor:
    """
    Monitors multiple symbols and stores predictions in database
    """
    
    def __init__(self, db_path: str = "/home/karthik/market.db", symbols_file: str = "/home/karthik/new/symbols.txt", monitor_db: str = "/home/karthik/monitoring.db"):
        self.db_path = db_path  # Main market data database
        self.monitor_db = monitor_db  # Separate database for monitoring data
        self.symbols_file = symbols_file
        self.predictor = OptimizedStockPredictor(db_path)
        
        # Load symbols
        self.symbols = self.load_symbols()
        self.logger = self.setup_logging()
        
        # Database tables
        self.create_tables()
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread = None
        self.outcome_thread = None
        
        # Strong signal criteria (updated per user requirements)
        self.strong_signal_criteria = {
            'min_confidence': 0.80,  # 80% confidence
            'min_profit_pct': 0.5,   # 0.5% profit potential
            'min_volume_ratio': 1.0  # No volume requirement
        }
        
        self.logger.info(f"Initialized monitor for {len(self.symbols)} symbols")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('multi_symbol_monitor.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def load_symbols(self) -> List[str]:
        """Load symbols from file"""
        try:
            with open(self.symbols_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            return symbols
        except Exception as e:
            print(f"Error loading symbols: {e}")
            return []
    
    def create_tables(self):
        """Create database tables for predictions and triggers"""
        try:
            with sqlite3.connect(self.monitor_db, timeout=30.0) as conn:
                # Predictions log table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    current_price REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    price_change REAL NOT NULL,
                    price_change_pct REAL NOT NULL,
                    confidence REAL NOT NULL,
                    signal TEXT NOT NULL,
                    action TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    potential_profit REAL NOT NULL,
                    potential_profit_pct REAL NOT NULL,
                    prediction_time REAL NOT NULL,
                    market_status TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Trading triggers table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    trigger_timestamp DATETIME NOT NULL,
                    current_price REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    target_minutes INTEGER NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    potential_profit_pct REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT DEFAULT 'monitoring',
                    actual_price_15min REAL,
                    outcome TEXT,
                    outcome_timestamp DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_logs_symbol_timestamp ON prediction_logs(symbol, timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp ON prediction_logs(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_trading_triggers_symbol_status ON trading_triggers(symbol, status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_trading_triggers_status ON trading_triggers(status)")
                
                self.logger.info("Database tables created successfully")
                
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
    
    def store_prediction(self, symbol: str, result: Dict, trading_signal: Dict, market_status: Dict):
        """Store prediction in database"""
        try:
            with sqlite3.connect(self.monitor_db, timeout=30.0) as conn:
                conn.execute("""
                INSERT INTO prediction_logs (
                    symbol, timestamp, current_price, predicted_price, price_change,
                    price_change_pct, confidence, signal, action, risk_level,
                    potential_profit, potential_profit_pct, prediction_time, market_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    result.get('prediction_time', datetime.now()),
                    result.get('current_price', 0.0),
                    result.get('predicted_price', 0.0),
                    result.get('price_change', 0.0),
                    result.get('price_change_pct', 0.0),
                    result.get('confidence', 0.0),
                    trading_signal.get('signal', 'HOLD'),
                    trading_signal.get('action', 'WAIT'),
                    trading_signal.get('risk_level', 'LOW'),
                    trading_signal.get('potential_profit', 0.0),
                    trading_signal.get('potential_profit_pct', 0.0),
                    result.get('total_prediction_time', 0.0),
                    market_status.get('status', 'UNKNOWN')
                ))
                
        except Exception as e:
            self.logger.error(f"Error storing prediction for {symbol}: {e}")
    
    def is_strong_signal(self, trading_signal: Dict, result: Dict) -> bool:
        """Check if signal meets strong signal criteria"""
        try:
            # Use confidence from result, not trading_signal
            confidence = result.get('confidence', 0.0)
            return (
                confidence >= self.strong_signal_criteria['min_confidence'] and
                trading_signal.get('potential_profit_pct', 0.0) >= self.strong_signal_criteria['min_profit_pct'] and
                trading_signal.get('signal', 'HOLD') in ['BUY', 'SELL']
            )
        except Exception as e:
            self.logger.error(f"Error checking strong signal: {e}")
            return False
    
    def create_trading_trigger(self, symbol: str, result: Dict, trading_signal: Dict, target_minutes: int):
        """Create trading trigger for strong signals"""
        try:
            with sqlite3.connect(self.monitor_db, timeout=30.0) as conn:
                conn.execute("""
                INSERT INTO trading_triggers (
                    symbol, trigger_timestamp, current_price, predicted_price,
                    target_minutes, signal, confidence, potential_profit_pct,
                    stop_loss, take_profit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    result.get('prediction_time', datetime.now()),
                    result.get('current_price', 0.0),
                    result.get('predicted_price', 0.0),
                    target_minutes,
                    trading_signal.get('signal', 'HOLD'),
                    result.get('confidence', 0.0),
                    trading_signal.get('potential_profit_pct', 0.0),
                    trading_signal.get('stop_loss'),
                    trading_signal.get('take_profit')
                ))
                
                self.logger.info(f"Created trading trigger for {symbol}: {trading_signal.get('signal', 'HOLD')} at ₹{result.get('current_price', 0.0):.2f}")
                
        except Exception as e:
            self.logger.error(f"Error creating trading trigger for {symbol}: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute("""
                SELECT ltp FROM ticks 
                WHERE symbol = ? 
                ORDER BY ts DESC 
                LIMIT 1
                """, (symbol,))
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    def update_trigger_outcomes(self):
        """Update outcomes for monitoring triggers with stop-loss checking"""
        try:
            with sqlite3.connect(self.monitor_db, timeout=30.0) as conn:
                # Get triggers that are still monitoring and past their target time
                cursor = conn.execute("""
                SELECT id, symbol, trigger_timestamp, current_price, predicted_price, 
                       target_minutes, signal, stop_loss, take_profit
                FROM trading_triggers 
                WHERE status = 'monitoring' 
                AND datetime(trigger_timestamp, '+' || target_minutes || ' minutes') <= datetime('now')
                """)
                
                triggers = cursor.fetchall()
                
                for trigger in triggers:
                    trigger_id, symbol, trigger_time, current_price, predicted_price, target_minutes, signal, stop_loss, take_profit = trigger
                    
                    # Get actual price at target time
                    actual_price = self.get_latest_price(symbol)
                    
                    if actual_price is not None:
                        # Check if stop loss was hit first (failure condition)
                        stop_loss_hit = False
                        if stop_loss is not None:
                            if signal == 'BUY' and actual_price <= stop_loss:
                                stop_loss_hit = True
                            elif signal == 'SELL' and actual_price >= stop_loss:
                                stop_loss_hit = True
                        
                        # Determine outcome based on user requirements
                        if stop_loss_hit:
                            # Stop loss hit = failure
                            outcome = 'failure'
                            self.logger.info(f"Stop loss hit for {symbol}: actual ₹{actual_price:.2f} vs stop loss ₹{stop_loss:.2f}")
                        else:
                            # Check if predicted price was reached
                            if signal == 'BUY':
                                # For BUY: success if actual price >= predicted price
                                outcome = 'success' if actual_price >= predicted_price else 'failure'
                            else:  # SELL
                                # For SELL: success if actual price <= predicted price
                                outcome = 'success' if actual_price <= predicted_price else 'failure'
                        
                        # Update trigger
                        conn.execute("""
                        UPDATE trading_triggers 
                        SET status = 'completed',
                            actual_price_15min = ?,
                            outcome = ?,
                            outcome_timestamp = datetime('now')
                        WHERE id = ?
                        """, (actual_price, outcome, trigger_id))
                        
                        self.logger.info(f"Updated trigger {trigger_id} for {symbol}: {outcome} (predicted: ₹{predicted_price:.2f}, actual: ₹{actual_price:.2f}, stop_loss: ₹{stop_loss:.2f})")
                
        except Exception as e:
            self.logger.error(f"Error updating trigger outcomes: {e}")
    
    def monitor_symbol(self, symbol: str, target_minutes: int = 15):
        """Monitor a single symbol"""
        try:
            # Load or train models
            if not self.predictor.load_models(symbol):
                self.logger.info(f"Training models for {symbol}")
                self.predictor.train_and_predict(symbol, target_minutes)
                return
            
            # Make prediction
            result = self.predictor.predict_fast(symbol, target_minutes)
            trading_signal = self.predictor.generate_trading_signal(result)
            market_status = self.predictor.get_market_status()
            
            # Store prediction
            self.store_prediction(symbol, result, trading_signal, market_status)
            
            # Check for strong signals
            if self.is_strong_signal(trading_signal, result):
                self.create_trading_trigger(symbol, result, trading_signal, target_minutes)
            
            self.logger.info(f"Monitored {symbol}: {trading_signal['signal']} (confidence: {result.get('confidence', 'N/A'):.1%})")
            
        except Exception as e:
            self.logger.error(f"Error monitoring {symbol}: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
    
    def monitor_all_symbols(self, target_minutes: int = 15):
        """Monitor all symbols"""
        self.logger.info(f"Starting monitoring cycle for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            try:
                self.monitor_symbol(symbol, target_minutes)
                time.sleep(0.1)  # Small delay between symbols
            except Exception as e:
                self.logger.error(f"Error in monitoring cycle for {symbol}: {e}")
        
        self.logger.info("Completed monitoring cycle")
    
    def monitoring_loop(self, target_minutes: int = 15, interval: int = 3):
        """Main monitoring loop - updated to run every 3 seconds"""
        self.logger.info(f"Starting monitoring loop (interval: {interval}s)")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Monitor all symbols
                self.monitor_all_symbols(target_minutes)
                
                # Update trigger outcomes
                self.update_trigger_outcomes()
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait 10 seconds on error (reduced from 60)
    
    def outcome_monitoring_loop(self):
        """Background loop for updating trigger outcomes"""
        while self.is_running:
            try:
                self.update_trigger_outcomes()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in outcome monitoring loop: {e}")
                time.sleep(60)
    
    def start_monitoring(self, target_minutes: int = 15, interval: int = 3):
        """Start monitoring all symbols - default 3 seconds interval"""
        if self.is_running:
            self.logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        
        # Start main monitoring thread
        self.monitor_thread = threading.Thread(
            target=self.monitoring_loop,
            args=(target_minutes, interval),
            daemon=True
        )
        self.monitor_thread.start()
        
        # Start outcome monitoring thread
        self.outcome_thread = threading.Thread(
            target=self.outcome_monitoring_loop,
            daemon=True
        )
        self.outcome_thread.start()
        
        self.logger.info("Multi-symbol monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()
        
        if self.outcome_thread and self.outcome_thread.is_alive():
            self.outcome_thread.join()
        
        self.logger.info("Multi-symbol monitoring stopped")
    
    def get_monitoring_stats(self) -> Dict:
        """Get monitoring statistics"""
        try:
            with sqlite3.connect(self.monitor_db, timeout=30.0) as conn:
                # Prediction stats
                cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT symbol) as symbols_monitored,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN signal = 'BUY' THEN 1 END) as buy_signals,
                    COUNT(CASE WHEN signal = 'SELL' THEN 1 END) as sell_signals,
                    COUNT(CASE WHEN signal = 'HOLD' THEN 1 END) as hold_signals
                FROM prediction_logs 
                WHERE timestamp >= datetime('now', '-1 hour')
                """)
                pred_stats = dict(cursor.fetchone())
                
                # Trigger stats
                cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_triggers,
                    COUNT(CASE WHEN status = 'monitoring' THEN 1 END) as active_triggers,
                    COUNT(CASE WHEN status = 'completed' AND outcome = 'success' THEN 1 END) as successful_triggers,
                    COUNT(CASE WHEN status = 'completed' AND outcome = 'failure' THEN 1 END) as failed_triggers
                FROM trading_triggers 
                WHERE trigger_timestamp >= datetime('now', '-24 hours')
                """)
                trigger_stats = dict(cursor.fetchone())
                
                return {
                    'predictions': pred_stats,
                    'triggers': trigger_stats,
                    'symbols_count': len(self.symbols),
                    'is_running': self.is_running
                }
                
        except Exception as e:
            self.logger.error(f"Error getting monitoring stats: {e}")
            return {}


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Symbol Stock Prediction Monitor')
    parser.add_argument('--minutes', type=int, default=15, help='Prediction timeframe in minutes')
    parser.add_argument('--interval', type=int, default=3, help='Monitoring interval in seconds (default: 3)')
    parser.add_argument('--stats', action='store_true', help='Show monitoring statistics')
    parser.add_argument('--once', action='store_true', help='Run monitoring once and exit')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = MultiSymbolMonitor()
    
    try:
        if args.stats:
            # Show statistics
            stats = monitor.get_monitoring_stats()
            print("\n" + "="*60)
            print("📊 MULTI-SYMBOL MONITORING STATISTICS")
            print("="*60)
            print(f"📈 Symbols Monitored: {stats.get('symbols_count', 0)}")
            print(f"🔄 Monitoring Status: {'RUNNING' if stats.get('is_running', False) else 'STOPPED'}")
            
            pred_stats = stats.get('predictions', {})
            print(f"\n📋 PREDICTIONS (Last Hour):")
            print(f"   Total Predictions: {pred_stats.get('total_predictions', 0)}")
            print(f"   Symbols Monitored: {pred_stats.get('symbols_monitored', 0)}")
            print(f"   Average Confidence: {pred_stats.get('avg_confidence', 0):.1%}")
            print(f"   BUY Signals: {pred_stats.get('buy_signals', 0)}")
            print(f"   SELL Signals: {pred_stats.get('sell_signals', 0)}")
            print(f"   HOLD Signals: {pred_stats.get('hold_signals', 0)}")
            
            trigger_stats = stats.get('triggers', {})
            print(f"\n🎯 TRADING TRIGGERS (Last 24 Hours):")
            print(f"   Total Triggers: {trigger_stats.get('total_triggers', 0)}")
            print(f"   Active Triggers: {trigger_stats.get('active_triggers', 0)}")
            print(f"   Successful: {trigger_stats.get('successful_triggers', 0)}")
            print(f"   Failed: {trigger_stats.get('failed_triggers', 0)}")
            
            success_rate = 0
            if trigger_stats.get('successful_triggers', 0) + trigger_stats.get('failed_triggers', 0) > 0:
                success_rate = trigger_stats.get('successful_triggers', 0) / (
                    trigger_stats.get('successful_triggers', 0) + trigger_stats.get('failed_triggers', 0)
                )
            print(f"   Success Rate: {success_rate:.1%}")
            print("="*60)
            
        elif args.once:
            # Run once
            print(f"Running monitoring cycle for {len(monitor.symbols)} symbols...")
            monitor.monitor_all_symbols(args.minutes)
            monitor.update_trigger_outcomes()
            print("Monitoring cycle completed")
            
        else:
            # Continuous monitoring
            print(f"Starting continuous monitoring for {len(monitor.symbols)} symbols")
            print(f"Prediction timeframe: {args.minutes} minutes")
            print(f"Monitoring interval: {args.interval} seconds (every 3 seconds for all symbols)")
            print("Press Ctrl+C to stop")
            
            monitor.start_monitoring(args.minutes, args.interval)
            
            try:
                while True:
                    time.sleep(30)  # Show stats every 30 seconds
                    # Show periodic stats
                    stats = monitor.get_monitoring_stats()
                    pred_stats = stats.get('predictions', {})
                    trigger_stats = stats.get('triggers', {})
                    
                    total_completed = trigger_stats.get('successful_triggers', 0) + trigger_stats.get('failed_triggers', 0)
                    success_rate = 0
                    if total_completed > 0:
                        success_rate = trigger_stats.get('successful_triggers', 0) / total_completed
                    
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Predictions: {pred_stats.get('total_predictions', 0)} | "
                          f"Active Triggers: {trigger_stats.get('active_triggers', 0)} | "
                          f"Success Rate: {success_rate:.1%} ({trigger_stats.get('successful_triggers', 0)}/{total_completed})")
                    
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                monitor.stop_monitoring()
                print("Monitoring stopped")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
