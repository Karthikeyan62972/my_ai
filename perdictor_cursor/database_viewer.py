#!/usr/bin/env python3
"""
Database Viewer for Multi-Symbol Monitoring
View predictions and trading triggers from database
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import argparse

class DatabaseViewer:
    """View and analyze prediction data from database"""
    
    def __init__(self, db_path: str = "/home/karthik/market.db", monitor_db: str = "/home/karthik/monitoring.db"):
        self.db_path = db_path  # Main market data
        self.monitor_db = monitor_db  # Monitoring data
    
    def get_recent_predictions(self, hours: int = 1, limit: int = 100):
        """Get recent predictions"""
        try:
            with sqlite3.connect(self.monitor_db, timeout=30.0) as conn:
                query = """
                SELECT symbol, timestamp, current_price, predicted_price, 
                       price_change_pct, confidence, signal, action, risk_level,
                       potential_profit_pct, market_status
                FROM prediction_logs 
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
                LIMIT ?
                """.format(hours)
                
                df = pd.read_sql_query(query, conn, params=(limit,))
                return df
        except Exception as e:
            print(f"Error getting recent predictions: {e}")
            return pd.DataFrame()
    
    def get_active_triggers(self):
        """Get active trading triggers"""
        try:
            with sqlite3.connect(self.monitor_db, timeout=30.0) as conn:
                query = """
                SELECT symbol, trigger_timestamp, current_price, predicted_price,
                       target_minutes, signal, confidence, potential_profit_pct,
                       stop_loss, take_profit, status
                FROM trading_triggers 
                WHERE status = 'monitoring'
                ORDER BY trigger_timestamp DESC
                """
                
                df = pd.read_sql_query(query, conn)
                return df
        except Exception as e:
            print(f"Error getting active triggers: {e}")
            return pd.DataFrame()
    
    def get_completed_triggers(self, days: int = 1, limit: int = 50):
        """Get completed trading triggers"""
        try:
            with sqlite3.connect(self.monitor_db, timeout=30.0) as conn:
                query = """
                SELECT symbol, trigger_timestamp, current_price, predicted_price,
                       actual_price_15min, signal, confidence, potential_profit_pct,
                       outcome, outcome_timestamp
                FROM trading_triggers 
                WHERE status = 'completed' 
                AND trigger_timestamp >= datetime('now', '-{} days')
                ORDER BY trigger_timestamp DESC
                LIMIT ?
                """.format(days)
                
                df = pd.read_sql_query(query, conn, params=(limit,))
                return df
        except Exception as e:
            print(f"Error getting completed triggers: {e}")
            return pd.DataFrame()
    
    def get_symbol_stats(self, symbol: str, hours: int = 24):
        """Get statistics for a specific symbol"""
        try:
            with sqlite3.connect(self.monitor_db, timeout=30.0) as conn:
                # Prediction stats
                pred_query = """
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN signal = 'BUY' THEN 1 END) as buy_signals,
                    COUNT(CASE WHEN signal = 'SELL' THEN 1 END) as sell_signals,
                    COUNT(CASE WHEN signal = 'HOLD' THEN 1 END) as hold_signals,
                    AVG(potential_profit_pct) as avg_profit_potential
                FROM prediction_logs 
                WHERE symbol = ? AND timestamp >= datetime('now', '-{} hours')
                """.format(hours)
                
                pred_stats = conn.execute(pred_query, (symbol,)).fetchone()
                
                # Trigger stats
                trigger_query = """
                SELECT 
                    COUNT(*) as total_triggers,
                    COUNT(CASE WHEN outcome = 'success' THEN 1 END) as successful,
                    COUNT(CASE WHEN outcome = 'failure' THEN 1 END) as failed
                FROM trading_triggers 
                WHERE symbol = ? AND trigger_timestamp >= datetime('now', '-{} days')
                """.format(hours // 24 + 1)
                
                trigger_stats = conn.execute(trigger_query, (symbol,)).fetchone()
                
                return {
                    'predictions': dict(zip([desc[0] for desc in conn.execute(pred_query, (symbol,)).description], pred_stats)),
                    'triggers': dict(zip([desc[0] for desc in conn.execute(trigger_query, (symbol,)).description], trigger_stats))
                }
        except Exception as e:
            print(f"Error getting symbol stats: {e}")
            return {}
    
    def get_overall_stats(self, hours: int = 24):
        """Get overall monitoring statistics"""
        try:
            with sqlite3.connect(self.monitor_db, timeout=30.0) as conn:
                # Overall prediction stats
                pred_query = """
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT symbol) as symbols_monitored,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN signal = 'BUY' THEN 1 END) as buy_signals,
                    COUNT(CASE WHEN signal = 'SELL' THEN 1 END) as sell_signals,
                    COUNT(CASE WHEN signal = 'HOLD' THEN 1 END) as hold_signals
                FROM prediction_logs 
                WHERE timestamp >= datetime('now', '-{} hours')
                """.format(hours)
                
                pred_stats = conn.execute(pred_query).fetchone()
                
                # Overall trigger stats
                trigger_query = """
                SELECT 
                    COUNT(*) as total_triggers,
                    COUNT(CASE WHEN status = 'monitoring' THEN 1 END) as active_triggers,
                    COUNT(CASE WHEN status = 'completed' AND outcome = 'success' THEN 1 END) as successful,
                    COUNT(CASE WHEN status = 'completed' AND outcome = 'failure' THEN 1 END) as failed
                FROM trading_triggers 
                WHERE trigger_timestamp >= datetime('now', '-{} days')
                """.format(hours // 24 + 1)
                
                trigger_stats = conn.execute(trigger_query).fetchone()
                
                return {
                    'predictions': dict(zip([desc[0] for desc in conn.execute(pred_query).description], pred_stats)),
                    'triggers': dict(zip([desc[0] for desc in conn.execute(trigger_query).description], trigger_stats))
                }
        except Exception as e:
            print(f"Error getting overall stats: {e}")
            return {}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Database Viewer for Multi-Symbol Monitoring')
    parser.add_argument('--recent', action='store_true', help='Show recent predictions')
    parser.add_argument('--active', action='store_true', help='Show active triggers')
    parser.add_argument('--completed', action='store_true', help='Show completed triggers')
    parser.add_argument('--stats', action='store_true', help='Show overall statistics')
    parser.add_argument('--symbol', type=str, help='Show stats for specific symbol')
    parser.add_argument('--hours', type=int, default=1, help='Hours to look back')
    parser.add_argument('--limit', type=int, default=20, help='Number of records to show')
    
    args = parser.parse_args()
    
    viewer = DatabaseViewer()
    
    if args.recent:
        print(f"\n📊 RECENT PREDICTIONS (Last {args.hours} hours)")
        print("="*80)
        df = viewer.get_recent_predictions(args.hours, args.limit)
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("No recent predictions found")
    
    elif args.active:
        print("\n🎯 ACTIVE TRADING TRIGGERS")
        print("="*80)
        df = viewer.get_active_triggers()
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("No active triggers found")
    
    elif args.completed:
        print(f"\n✅ COMPLETED TRADING TRIGGERS (Last {args.hours//24 + 1} days)")
        print("="*80)
        df = viewer.get_completed_triggers(args.hours//24 + 1, args.limit)
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("No completed triggers found")
    
    elif args.symbol:
        print(f"\n📈 STATISTICS FOR {args.symbol}")
        print("="*60)
        stats = viewer.get_symbol_stats(args.symbol, args.hours)
        if stats:
            pred_stats = stats.get('predictions', {})
            trigger_stats = stats.get('triggers', {})
            
            print(f"📊 PREDICTIONS (Last {args.hours} hours):")
            print(f"   Total: {pred_stats.get('total_predictions', 0)}")
            print(f"   Avg Confidence: {pred_stats.get('avg_confidence', 0):.1%}")
            print(f"   BUY: {pred_stats.get('buy_signals', 0)}")
            print(f"   SELL: {pred_stats.get('sell_signals', 0)}")
            print(f"   HOLD: {pred_stats.get('hold_signals', 0)}")
            print(f"   Avg Profit Potential: {pred_stats.get('avg_profit_potential', 0):.2f}%")
            
            print(f"\n🎯 TRIGGERS:")
            print(f"   Total: {trigger_stats.get('total_triggers', 0)}")
            print(f"   Successful: {trigger_stats.get('successful', 0)}")
            print(f"   Failed: {trigger_stats.get('failed', 0)}")
            
            success_rate = 0
            if trigger_stats.get('successful', 0) + trigger_stats.get('failed', 0) > 0:
                success_rate = trigger_stats.get('successful', 0) / (
                    trigger_stats.get('successful', 0) + trigger_stats.get('failed', 0)
                )
            print(f"   Success Rate: {success_rate:.1%}")
        else:
            print("No data found for this symbol")
    
    elif args.stats:
        print(f"\n📊 OVERALL MONITORING STATISTICS (Last {args.hours} hours)")
        print("="*60)
        stats = viewer.get_overall_stats(args.hours)
        if stats:
            pred_stats = stats.get('predictions', {})
            trigger_stats = stats.get('triggers', {})
            
            print(f"📈 PREDICTIONS:")
            print(f"   Total: {pred_stats.get('total_predictions', 0)}")
            print(f"   Symbols: {pred_stats.get('symbols_monitored', 0)}")
            print(f"   Avg Confidence: {pred_stats.get('avg_confidence', 0):.1%}")
            print(f"   BUY: {pred_stats.get('buy_signals', 0)}")
            print(f"   SELL: {pred_stats.get('sell_signals', 0)}")
            print(f"   HOLD: {pred_stats.get('hold_signals', 0)}")
            
            print(f"\n🎯 TRADING TRIGGERS:")
            print(f"   Total: {trigger_stats.get('total_triggers', 0)}")
            print(f"   Active: {trigger_stats.get('active_triggers', 0)}")
            print(f"   Successful: {trigger_stats.get('successful', 0)}")
            print(f"   Failed: {trigger_stats.get('failed', 0)}")
            
            success_rate = 0
            if trigger_stats.get('successful', 0) + trigger_stats.get('failed', 0) > 0:
                success_rate = trigger_stats.get('successful', 0) / (
                    trigger_stats.get('successful', 0) + trigger_stats.get('failed', 0)
                )
            print(f"   Success Rate: {success_rate:.1%}")
        else:
            print("No statistics available")
    
    else:
        print("Please specify an option: --recent, --active, --completed, --stats, or --symbol SYMBOL")


if __name__ == "__main__":
    main()
