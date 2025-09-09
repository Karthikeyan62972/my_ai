#!/usr/bin/env python3
"""
Multi-Symbol Monitor - Using the robust predictor implementation
"""

import argparse
import time
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from optimized_predictor_v2 import RobustStockPredictor


class MultiSymbolMonitor:
    """Monitor multiple symbols with robust prediction system"""
    
    def __init__(self, db_path: str, symbols_file: str, monitoring_db: str = "/home/karthik/monitoring.db"):
        self.db_path = db_path
        self.symbols_file = symbols_file
        self.monitoring_db = monitoring_db
        
        # Setup logging first
        self.setup_logging()
        
        # Initialize predictor
        self.predictor = RobustStockPredictor(db_path)
        
        # Load symbols
        self.symbols = self._load_symbols()
        
        # Setup monitoring database
        self._setup_monitoring_db()
        
        # Strong signal criteria
        self.strong_signal_criteria = {
            'min_confidence': 0.8,  # 80% confidence
            'min_profit_pct': 0.5   # 0.5% profit
        }
        
        self.logger.info(f"Initialized monitor for {len(self.symbols)} symbols")
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('multi_symbol_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_symbols(self) -> List[str]:
        """Load symbols from file"""
        try:
            with open(self.symbols_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            return symbols
        except Exception as e:
            self.logger.error(f"Error loading symbols: {e}")
            return []
    
    def _setup_monitoring_db(self):
        """Setup monitoring database"""
        try:
            with sqlite3.connect(self.monitoring_db, timeout=30.0) as conn:
                # Predictions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        current_price REAL NOT NULL,
                        predicted_price REAL NOT NULL,
                        confidence REAL NOT NULL,
                        price_change REAL NOT NULL,
                        price_change_pct REAL NOT NULL,
                        signal TEXT NOT NULL,
                        target_minutes INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
                
                # Trading triggers table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trading_triggers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        trigger_time TEXT NOT NULL,
                        current_price REAL NOT NULL,
                        predicted_price REAL NOT NULL,
                        confidence REAL NOT NULL,
                        potential_profit_pct REAL NOT NULL,
                        signal TEXT NOT NULL,
                        action TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        target_minutes INTEGER NOT NULL,
                        status TEXT NOT NULL DEFAULT 'monitoring',
                        outcome TEXT,
                        completion_time TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                
                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time ON predictions(symbol, timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_triggers_symbol_status ON trading_triggers(symbol, status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_triggers_status ON trading_triggers(status)")
                
                conn.commit()
                self.logger.info("Monitoring database setup completed")
                
        except Exception as e:
            self.logger.error(f"Error setting up monitoring database: {e}")
            raise
    
    def _store_prediction(self, result: Dict[str, Any], signal: Dict[str, Any]):
        """Store prediction in database"""
        try:
            with sqlite3.connect(self.monitoring_db, timeout=30.0) as conn:
                conn.execute("""
                    INSERT INTO prediction_logs 
                    (symbol, timestamp, current_price, predicted_price, price_change, 
                     price_change_pct, confidence, signal, action, risk_level, 
                     potential_profit, potential_profit_pct, prediction_time, market_status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result['symbol'],
                    result['timestamp'],
                    result['current_price'],
                    result['predicted_price'],
                    result['price_change'],
                    result['price_change_pct'],
                    result['confidence'],
                    signal['signal'],
                    signal.get('action', 'HOLD'),
                    signal.get('risk_level', 'LOW'),
                    signal.get('potential_profit', 0.0),
                    signal.get('potential_profit_pct', 0.0),
                    result['target_minutes'],
                    'LIVE',
                    datetime.now().isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing prediction: {e}")
    
    def _create_trading_trigger(self, result: Dict[str, Any], signal: Dict[str, Any]):
        """Create trading trigger if conditions are met"""
        try:
            confidence = signal['confidence']
            profit_pct = signal['potential_profit_pct']
            
            # Check if this is a strong signal
            if (confidence >= self.strong_signal_criteria['min_confidence'] and 
                profit_pct >= self.strong_signal_criteria['min_profit_pct']):
                
                with sqlite3.connect(self.monitoring_db, timeout=30.0) as conn:
                    conn.execute("""
                        INSERT INTO trading_triggers 
                        (symbol, trigger_time, current_price, predicted_price, confidence,
                         potential_profit_pct, signal, action, risk_level, target_minutes, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        result['symbol'],
                        result['timestamp'],
                        result['current_price'],
                        result['predicted_price'],
                        confidence,
                        profit_pct,
                        signal['signal'],
                        signal['action'],
                        signal['risk_level'],
                        result['target_minutes'],
                        datetime.now().isoformat()
                    ))
                    conn.commit()
                    
                self.logger.info(f"🎯 STRONG SIGNAL: {result['symbol']} - {signal['signal']} "
                               f"(Confidence: {confidence:.1%}, Profit: {profit_pct:.2f}%)")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error creating trading trigger: {e}")
        
        return False
    
    def _update_trigger_outcomes(self):
        """Update outcomes for completed triggers"""
        try:
            with sqlite3.connect(self.monitoring_db, timeout=30.0) as conn:
                # Get monitoring triggers
                cursor = conn.execute("""
                    SELECT id, symbol, trigger_time, predicted_price, signal, target_minutes
                    FROM trading_triggers 
                    WHERE status = 'monitoring'
                """)
                
                triggers = cursor.fetchall()
                
                for trigger in triggers:
                    trigger_id, symbol, trigger_time, predicted_price, signal, target_minutes = trigger
                    
                    # Check if trigger time has passed
                    trigger_dt = datetime.fromisoformat(trigger_time)
                    completion_time = trigger_dt.timestamp() + (target_minutes * 60)
                    
                    if time.time() >= completion_time:
                        # Get actual price at completion time
                        try:
                            actual_price = self._get_price_at_time(symbol, completion_time)
                            
                            if actual_price is not None:
                                # Determine outcome
                                if signal == 'BUY':
                                    outcome = 'success' if actual_price >= predicted_price else 'failure'
                                else:  # SELL
                                    outcome = 'success' if actual_price <= predicted_price else 'failure'
                                
                                # Update trigger
                                conn.execute("""
                                    UPDATE trading_triggers 
                                    SET status = 'completed', outcome = ?, completion_time = ?
                                    WHERE id = ?
                                """, (outcome, datetime.fromtimestamp(completion_time).isoformat(), trigger_id))
                                
                                self.logger.info(f"✅ Trigger {trigger_id} completed: {outcome} "
                                               f"(Predicted: {predicted_price:.2f}, Actual: {actual_price:.2f})")
                            
                        except Exception as e:
                            self.logger.error(f"Error updating trigger {trigger_id}: {e}")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating trigger outcomes: {e}")
    
    def _get_price_at_time(self, symbol: str, timestamp: float) -> float:
        """Get price at specific timestamp"""
        try:
            target_time = datetime.fromtimestamp(timestamp).isoformat()
            
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute("""
                    SELECT ltp FROM ticks 
                    WHERE symbol = ? AND ts <= ? 
                    ORDER BY ts DESC LIMIT 1
                """, (symbol, target_time))
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            self.logger.error(f"Error getting price at time: {e}")
            return None
    
    def monitor_symbol(self, symbol: str, target_minutes: int = 15) -> bool:
        """Monitor a single symbol"""
        try:
            # Make prediction
            result = self.predictor.predict(symbol, target_minutes)
            signal = self.predictor.generate_trading_signal(result)
            
            # Store prediction
            self._store_prediction(result, signal)
            
            # Check for strong signal
            trigger_created = self._create_trading_trigger(result, signal)
            
            # Log result
            status = "🎯 TRIGGER" if trigger_created else "📊 MONITOR"
            self.logger.info(f"{status} {symbol}: {signal['signal']} "
                           f"(Price: {result['current_price']:.2f} → {result['predicted_price']:.2f}, "
                           f"Conf: {result['confidence']:.1%}, Profit: {signal['potential_profit_pct']:.2f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error monitoring {symbol}: {e}")
            return False
    
    def run_monitoring_cycle(self, target_minutes: int = 15):
        """Run one monitoring cycle for all symbols"""
        self.logger.info(f"🔄 Starting monitoring cycle for {len(self.symbols)} symbols")
        
        successful = 0
        failed = 0
        
        for symbol in self.symbols:
            if self.monitor_symbol(symbol, target_minutes):
                successful += 1
            else:
                failed += 1
        
        # Update trigger outcomes
        self._update_trigger_outcomes()
        
        self.logger.info(f"✅ Monitoring cycle completed: {successful} successful, {failed} failed")
    
    def run_continuous_monitoring(self, target_minutes: int = 15, interval: int = 3):
        """Run continuous monitoring"""
        self.logger.info(f"🚀 Starting continuous monitoring (interval: {interval}s, target: {target_minutes}min)")
        
        try:
            while True:
                cycle_start = time.time()
                self.run_monitoring_cycle(target_minutes)
                
                # Calculate sleep time
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, interval - cycle_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            self.logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Error in continuous monitoring: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-Symbol Stock Monitor')
    parser.add_argument('--minutes', type=int, default=15, help='Prediction target minutes')
    parser.add_argument('--interval', type=int, default=3, help='Monitoring interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run once instead of continuous')
    parser.add_argument('--symbols', type=str, default='/home/karthik/new/symbols.txt', help='Symbols file path')
    parser.add_argument('--db', type=str, default='/home/karthik/market.db', help='Market database path')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = MultiSymbolMonitor(args.db, args.symbols)
    
    if args.once:
        monitor.run_monitoring_cycle(args.minutes)
    else:
        monitor.run_continuous_monitoring(args.minutes, args.interval)


if __name__ == "__main__":
    main()
