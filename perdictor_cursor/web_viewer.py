#!/usr/bin/env python3
"""
Web-based Database Viewer for Stock Prediction Monitoring
Beautiful, modern UI for viewing predictions and trading triggers
"""

from flask import Flask, render_template, jsonify, request
import sqlite3
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional

app = Flask(__name__)

# Configuration
MONITOR_DB = '/home/karthik/monitoring.db'
MARKET_DB = '/home/karthik/market.db'

class WebDatabaseViewer:
    def __init__(self, monitor_db: str, market_db: str):
        self.monitor_db = monitor_db
        self.market_db = market_db
    
    def get_connection(self, db_path: str):
        """Get database connection with timeout"""
        return sqlite3.connect(db_path, timeout=30.0)
    
    def get_overall_stats(self, hours: int = 24) -> Dict:
        """Get overall monitoring statistics"""
        try:
            with self.get_connection(self.monitor_db) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get prediction stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        COUNT(DISTINCT symbol) as symbols_monitored,
                        AVG(confidence) as avg_confidence,
                        COUNT(CASE WHEN signal = 'BUY' THEN 1 END) as buy_signals,
                        COUNT(CASE WHEN signal = 'SELL' THEN 1 END) as sell_signals,
                        COUNT(CASE WHEN signal = 'HOLD' THEN 1 END) as hold_signals,
                        COUNT(CASE WHEN signal = 'AVOID' THEN 1 END) as avoid_signals
                    FROM prediction_logs 
                    WHERE timestamp >= datetime('now', '-{} hours')
                """.format(hours))
                
                pred_stats = dict(cursor.fetchone())
                
                # Get trigger stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_triggers,
                        COUNT(CASE WHEN status = 'monitoring' THEN 1 END) as active_triggers,
                        COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_triggers,
                        COUNT(CASE WHEN status = 'failure' THEN 1 END) as failed_triggers
                    FROM trading_triggers
                    WHERE trigger_timestamp >= datetime('now', '-{} hours')
                """.format(hours))
                
                trigger_stats = dict(cursor.fetchone())
                
                # Calculate success rate
                total_completed = trigger_stats['successful_triggers'] + trigger_stats['failed_triggers']
                success_rate = (trigger_stats['successful_triggers'] / total_completed * 100) if total_completed > 0 else 0
                
                return {
                    'predictions': pred_stats,
                    'triggers': trigger_stats,
                    'success_rate': round(success_rate, 1),
                    'hours': hours
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def get_recent_predictions(self, hours: int = 1, limit: int = 50) -> List[Dict]:
        """Get recent predictions"""
        try:
            with self.get_connection(self.monitor_db) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        symbol, timestamp, current_price, predicted_price, 
                        price_change, price_change_pct, confidence, signal, 
                        action, risk_level, potential_profit, potential_profit_pct,
                        prediction_time, market_status
                    FROM prediction_logs 
                    WHERE timestamp >= datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                    LIMIT ?
                """.format(hours), (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            return [{'error': str(e)}]
    
    def get_active_triggers(self) -> List[Dict]:
        """Get active trading triggers"""
        try:
            with self.get_connection(self.monitor_db) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        id, symbol, trigger_timestamp, current_price, predicted_price,
                        target_minutes, signal, confidence, potential_profit_pct,
                        stop_loss, take_profit, status, outcome_price, outcome_timestamp
                    FROM trading_triggers 
                    WHERE status = 'monitoring'
                    ORDER BY trigger_timestamp DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            return [{'error': str(e)}]
    
    def get_symbol_stats(self, symbol: str, hours: int = 24) -> Dict:
        """Get statistics for a specific symbol"""
        try:
            with self.get_connection(self.monitor_db) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        AVG(confidence) as avg_confidence,
                        AVG(price_change_pct) as avg_price_change_pct,
                        COUNT(CASE WHEN signal = 'BUY' THEN 1 END) as buy_signals,
                        COUNT(CASE WHEN signal = 'SELL' THEN 1 END) as sell_signals,
                        COUNT(CASE WHEN signal = 'HOLD' THEN 1 END) as hold_signals,
                        MAX(current_price) as max_price,
                        MIN(current_price) as min_price,
                        AVG(current_price) as avg_price
                    FROM prediction_logs 
                    WHERE symbol = ? AND timestamp >= datetime('now', '-{} hours')
                """.format(hours), (symbol,))
                
                stats = dict(cursor.fetchone())
                
                # Get latest prediction
                cursor.execute("""
                    SELECT * FROM prediction_logs 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (symbol,))
                
                latest = dict(cursor.fetchone()) if cursor.fetchone() else None
                
                return {
                    'symbol': symbol,
                    'stats': stats,
                    'latest': latest
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def get_symbols_list(self) -> List[str]:
        """Get list of all monitored symbols"""
        try:
            with self.get_connection(self.monitor_db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT symbol FROM prediction_logs ORDER BY symbol")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            return []

# Initialize viewer
viewer = WebDatabaseViewer(MONITOR_DB, MARKET_DB)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def api_stats():
    """API endpoint for overall statistics"""
    hours = request.args.get('hours', 24, type=int)
    return jsonify(viewer.get_overall_stats(hours))

@app.route('/api/predictions')
def api_predictions():
    """API endpoint for recent predictions"""
    hours = request.args.get('hours', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    symbol = request.args.get('symbol', None)
    
    predictions = viewer.get_recent_predictions(hours, limit)
    
    if symbol:
        predictions = [p for p in predictions if p.get('symbol') == symbol]
    
    return jsonify(predictions)

@app.route('/api/triggers')
def api_triggers():
    """API endpoint for active triggers"""
    return jsonify(viewer.get_active_triggers())

@app.route('/api/symbols')
def api_symbols():
    """API endpoint for symbols list"""
    return jsonify(viewer.get_symbols_list())

@app.route('/api/symbol/<symbol>')
def api_symbol_stats(symbol):
    """API endpoint for symbol-specific statistics"""
    hours = request.args.get('hours', 24, type=int)
    return jsonify(viewer.get_symbol_stats(symbol, hours))

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting Web Database Viewer...")
    print("📊 Dashboard: http://localhost:5001")
    print("🔗 API Endpoints:")
    print("   - /api/stats - Overall statistics")
    print("   - /api/predictions - Recent predictions")
    print("   - /api/triggers - Active triggers")
    print("   - /api/symbols - Symbols list")
    print("   - /api/symbol/<symbol> - Symbol stats")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
