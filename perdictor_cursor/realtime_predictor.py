#!/usr/bin/env python3
"""
Real-Time Stock Price Predictor for Live Trading
Optimized for speed and continuous learning during trading hours
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
import os
from pathlib import Path
import threading
import time
from time import perf_counter

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Optional advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')

class RealTimeStockPredictor:
    """
    Real-time stock price predictor optimized for live trading
    Features:
    - Fast prediction (< 1 second)
    - Incremental learning
    - Real-time data updates
    - Model adaptation
    """
    
    def __init__(self, db_path: str = "/home/karthik/market.db", model_dir: str = "realtime_models"):
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Lightweight models for speed
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=50,  # Reduced for speed
                max_depth=6,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            ),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0)
        }
        
        # Add XGBoost if available (lightweight version)
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=50,  # Reduced for speed
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.last_update = {}
        self.update_interval = 300  # 5 minutes
        
        # Setup logging
        self.setup_logging()
        
        # Background update thread
        self.update_thread = None
        self.stop_updates = False
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('realtime_predictor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def connect_db(self) -> sqlite3.Connection:
        """Create database connection with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}")
            raise
    
    def fetch_recent_data(self, symbol: str, minutes: int = 60) -> pd.DataFrame:
        """
        Fetch recent data for real-time prediction
        
        Args:
            symbol: Stock symbol
            minutes: Minutes of recent data to fetch
            
        Returns:
            DataFrame with recent data
        """
        db_start_time = perf_counter()
        
        # Calculate timestamp for recent data
        current_time = datetime.now()
        start_time = current_time - timedelta(minutes=minutes)
        start_timestamp = int(start_time.timestamp())
        
        query = """
        SELECT symbol,
               ts,
               ltp,
               bid,
               ask,
               bid_qty,
               ask_qty,
               vwap,
               vol_traded_today,
               last_traded_qty,
               exch_feed_time,
               bid_size,
               ask_size,
               bid_price,
               ask_price,
               tot_buy_qty,
               tot_sell_qty,
               avg_trade_price,
               ch,
               chp,
               last_traded_time,
               high,
               low,
               open,
               prev_close,
               high_price,
               low_price,
               open_price,
               prev_close_price
        FROM ticks
        WHERE symbol = ? AND ts >= ?
        ORDER BY ts DESC
        LIMIT 1000
        """
        
        try:
            with self.connect_db() as conn:
                df = pd.read_sql_query(query, conn, params=(symbol, start_timestamp))
                
            if df.empty:
                # Fallback to last 1000 records if no recent data
                return self.fetch_data(symbol, limit=1000)
                
            # Convert timestamp columns
            df['ts'] = pd.to_datetime(df['ts'], unit='s')
            df['exch_feed_time'] = pd.to_datetime(df['exch_feed_time'], unit='s')
            df['last_traded_time'] = pd.to_datetime(df['last_traded_time'], unit='s')
            
            # Sort by timestamp (oldest first for proper time series)
            df = df.sort_values('ts').reset_index(drop=True)
            
            db_fetch_time = perf_counter() - db_start_time
            self.logger.info(f"Fetched {len(df)} recent records for {symbol} in {db_fetch_time:.3f} seconds")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching recent data for {symbol}: {e}")
            raise
    
    def fetch_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch historical data (fallback method)"""
        query = """
        SELECT symbol,
               ts,
               ltp,
               bid,
               ask,
               bid_qty,
               ask_qty,
               vwap,
               vol_traded_today,
               last_traded_qty,
               exch_feed_time,
               bid_size,
               ask_size,
               bid_price,
               ask_price,
               tot_buy_qty,
               tot_sell_qty,
               avg_trade_price,
               ch,
               chp,
               last_traded_time,
               high,
               low,
               open,
               prev_close,
               high_price,
               low_price,
               open_price,
               prev_close_price
        FROM ticks
        WHERE symbol = ?
        ORDER BY ts DESC
        LIMIT ?
        """
        
        try:
            with self.connect_db() as conn:
                df = pd.read_sql_query(query, conn, params=(symbol, limit))
                
            if df.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
                
            # Convert timestamp columns
            df['ts'] = pd.to_datetime(df['ts'], unit='s')
            df['exch_feed_time'] = pd.to_datetime(df['exch_feed_time'], unit='s')
            df['last_traded_time'] = pd.to_datetime(df['last_traded_time'], unit='s')
            
            # Sort by timestamp (oldest first for proper time series)
            df = df.sort_values('ts').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    def create_fast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create essential features quickly for real-time prediction
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with essential features
        """
        df = df.copy()
        
        # Essential price features
        df['price_change'] = df['ltp'].diff()
        df['price_change_pct'] = df['ltp'].pct_change()
        df['bid_ask_spread'] = df['ask'] - df['bid']
        df['bid_ask_spread_pct'] = (df['ask'] - df['bid']) / df['ltp']
        
        # Volume features
        df['volume_change'] = df['vol_traded_today'].diff()
        df['volume_change_pct'] = df['vol_traded_today'].pct_change()
        df['buy_sell_ratio'] = df['tot_buy_qty'] / (df['tot_sell_qty'] + 1e-8)
        
        # Price position features
        df['price_vs_high'] = df['ltp'] / df['high']
        df['price_vs_low'] = df['ltp'] / df['low']
        df['price_vs_open'] = df['ltp'] / df['open']
        df['price_vs_prev_close'] = df['ltp'] / df['prev_close']
        
        # Simple moving averages (fast)
        df['sma_5'] = df['ltp'].rolling(window=5).mean()
        df['sma_10'] = df['ltp'].rolling(window=10).mean()
        df['ema_5'] = df['ltp'].ewm(span=5).mean()
        
        # Simple RSI (fast)
        delta = df['ltp'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Lag features (essential ones only)
        for lag in [1, 2, 3]:
            df[f'ltp_lag_{lag}'] = df['ltp'].shift(lag)
            df[f'volume_lag_{lag}'] = df['vol_traded_today'].shift(lag)
        
        # Rolling statistics (fast)
        df['ltp_std_5'] = df['ltp'].rolling(window=5).std()
        df['ltp_mean_5'] = df['ltp'].rolling(window=5).mean()
        
        # Time-based features
        df['hour'] = df['ts'].dt.hour
        df['minute'] = df['ts'].dt.minute
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, target_minutes: int = 15) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with target variable"""
        # Create target variable (price after target_minutes)
        df['target'] = df['ltp'].shift(-target_minutes)
        
        # Remove rows with NaN values in target
        df_clean = df.dropna(subset=['target'])
        
        # Fill remaining NaN values
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        if len(df_clean) < 50:
            raise ValueError(f"Insufficient data for training: {len(df_clean)} records")
        
        # Select feature columns (exclude target and metadata)
        exclude_cols = ['symbol', 'ts', 'exch_feed_time', 'last_traded_time', 'target', 'ltp']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        # Remove columns with too many NaN values or constant values
        valid_feature_cols = []
        for col in feature_cols:
            if df_clean[col].isna().sum() < len(df_clean) * 0.5:
                if df_clean[col].nunique() > 1:
                    valid_feature_cols.append(col)
        
        self.feature_columns = valid_feature_cols
        X = df_clean[valid_feature_cols]
        y = df_clean['target']
        
        self.logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_models_fast(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> Dict[str, float]:
        """Fast model training for real-time use"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        scores = {}
        
        for name, model in self.models.items():
            try:
                # Quick training without cross-validation for speed
                model.fit(X_scaled, y)
                
                # Simple validation score
                pred = model.predict(X_scaled)
                mae = mean_absolute_error(y, pred)
                scores[name] = mae
                
                self.logger.info(f"Trained {name} model with MAE: {mae:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name} model: {e}")
                scores[name] = float('inf')
        
        # Save models
        symbol_clean = symbol.replace(':', '_').replace('-', '_')
        symbol_model_dir = self.model_dir / symbol_clean
        symbol_model_dir.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, symbol_model_dir / f"{name}_model.pkl")
        joblib.dump(self.scaler, symbol_model_dir / "scaler.pkl")
        joblib.dump(self.feature_columns, symbol_model_dir / "feature_columns.pkl")
        
        self.is_trained = True
        return scores
    
    def load_models(self, symbol: str) -> bool:
        """Load pre-trained models for a specific symbol"""
        try:
            symbol_clean = symbol.replace(':', '_').replace('-', '_')
            model_dir = self.model_dir / symbol_clean
            
            if not model_dir.exists():
                return False
                
            for name in self.models.keys():
                model_path = model_dir / f"{name}_model.pkl"
                if model_path.exists():
                    self.models[name] = joblib.load(model_path)
                else:
                    return False
            
            self.scaler = joblib.load(model_dir / "scaler.pkl")
            self.feature_columns = joblib.load(model_dir / "feature_columns.pkl")
            self.is_trained = True
            
            self.logger.info(f"Loaded models for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def predict_fast(self, symbol: str, target_minutes: int = 15) -> Dict:
        """
        Fast prediction for real-time trading
        
        Args:
            symbol: Stock symbol to predict
            target_minutes: Minutes ahead to predict
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Please train models first.")
        
        try:
            prediction_start = perf_counter()
            
            # Fetch recent data (last 60 minutes)
            df = self.fetch_recent_data(symbol, minutes=60)
            
            # Create features
            df_features = self.create_fast_features(df)
            
            # Get latest data point
            latest_data = df_features.iloc[-1:].copy()
            
            # Prepare features for prediction
            X_pred = latest_data[self.feature_columns].fillna(0)
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Make predictions with all models
            predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_pred_scaled)[0]
                    predictions[name] = pred
                except Exception as e:
                    self.logger.error(f"Error predicting with {name}: {e}")
                    predictions[name] = None
            
            # Calculate ensemble prediction (weighted average)
            valid_predictions = {k: v for k, v in predictions.items() if v is not None}
            if valid_predictions:
                # Weight models based on performance
                model_weights = {
                    'xgboost': 0.4,
                    'random_forest': 0.4,
                    'linear_regression': 0.1,
                    'ridge': 0.1
                }
                
                weighted_sum = 0
                total_weight = 0
                for model, pred in valid_predictions.items():
                    weight = model_weights.get(model, 0.1)
                    weighted_sum += pred * weight
                    total_weight += weight
                
                if total_weight > 0:
                    ensemble_pred = weighted_sum / total_weight
                else:
                    ensemble_pred = np.mean(list(valid_predictions.values()))
            else:
                raise ValueError("No valid predictions available")
            
            # Get current price and time
            current_price = latest_data['ltp'].iloc[0]
            last_trade_time = latest_data['last_traded_time'].iloc[0]
            
            # Apply realistic bounds
            ensemble_pred = self._apply_realistic_bounds(ensemble_pred, current_price, target_minutes)
            
            total_time = perf_counter() - prediction_start
            
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': ensemble_pred,
                'price_change': ensemble_pred - current_price,
                'price_change_pct': ((ensemble_pred - current_price) / current_price) * 100,
                'last_trade_time': last_trade_time,
                'prediction_time': datetime.now(),
                'target_minutes': target_minutes,
                'individual_predictions': predictions,
                'confidence': self._calculate_confidence(predictions, current_price),
                'prediction_speed': 'fast',
                'total_prediction_time': total_time
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {e}")
            raise
    
    def _apply_realistic_bounds(self, predicted_price: float, current_price: float, target_minutes: int) -> float:
        """Apply realistic bounds to predictions"""
        max_change_pct = min(2.0, target_minutes * 0.1)
        max_change = current_price * (max_change_pct / 100)
        
        min_price = current_price - max_change
        max_price = current_price + max_change
        
        bounded_price = max(min_price, min(max_price, predicted_price))
        
        if bounded_price != predicted_price:
            self.logger.warning(f"Applied realistic bounds: {predicted_price:.2f} -> {bounded_price:.2f}")
        
        return bounded_price
    
    def _calculate_confidence(self, predictions: Dict, current_price: float) -> float:
        """Calculate prediction confidence based on model agreement"""
        valid_preds = [v for v in predictions.values() if v is not None]
        if len(valid_preds) < 2:
            return 0.5
        
        mean_pred = np.mean(valid_preds)
        std_pred = np.std(valid_preds)
        cv = std_pred / mean_pred if mean_pred != 0 else 1
        
        confidence = max(0, min(1, 1 - cv))
        return confidence
    
    def incremental_update(self, symbol: str):
        """Incremental model update with recent data"""
        try:
            # Fetch recent data
            df = self.fetch_recent_data(symbol, minutes=30)
            
            if len(df) < 100:
                self.logger.warning(f"Insufficient recent data for {symbol}: {len(df)} records")
                return
            
            # Create features
            df_features = self.create_fast_features(df)
            
            # Prepare training data
            X, y = self.prepare_training_data(df_features, 15)
            
            # Quick retrain with recent data
            self.train_models_fast(X, y, symbol)
            
            self.last_update[symbol] = datetime.now()
            self.logger.info(f"Incremental update completed for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error in incremental update for {symbol}: {e}")
    
    def start_background_updates(self):
        """Start background thread for model updates"""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.stop_updates = False
            self.update_thread = threading.Thread(target=self._background_update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            self.logger.info("Started background update thread")
    
    def stop_background_updates(self):
        """Stop background update thread"""
        self.stop_updates = True
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join()
        self.logger.info("Stopped background update thread")
    
    def _background_update_loop(self):
        """Background loop for model updates"""
        while not self.stop_updates:
            try:
                # Update models for symbols that need updating
                for symbol in self.last_update.keys():
                    if (datetime.now() - self.last_update[symbol]).seconds > self.update_interval:
                        self.incremental_update(symbol)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in background update loop: {e}")
                time.sleep(60)
    
    def train_and_predict(self, symbol: str, target_minutes: int = 15) -> Dict:
        """Train models and make prediction for a symbol"""
        try:
            self.logger.info(f"Starting fast training and prediction for {symbol}")
            
            # Fetch data
            df = self.fetch_data(symbol, limit=2000)  # Reduced for speed
            
            # Create features
            df_features = self.create_fast_features(df)
            
            # Prepare training data
            X, y = self.prepare_training_data(df_features, target_minutes)
            
            # Train models
            scores = self.train_models_fast(X, y, symbol)
            self.logger.info(f"Fast training completed. Scores: {scores}")
            
            # Make prediction
            result = self.predict_fast(symbol, target_minutes)
            
            # Record last update
            self.last_update[symbol] = datetime.now()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in train_and_predict for {symbol}: {e}")
            raise


def main():
    """Main function for real-time predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Stock Price Predictor')
    parser.add_argument('symbol', help='Stock symbol to predict')
    parser.add_argument('--minutes', type=int, default=15, help='Minutes ahead to predict')
    parser.add_argument('--retrain', action='store_true', help='Force retrain models')
    parser.add_argument('--continuous', action='store_true', help='Run continuous predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = RealTimeStockPredictor()
    
    try:
        if args.continuous:
            # Continuous mode for live trading
            print(f"Starting continuous predictions for {args.symbol}")
            predictor.start_background_updates()
            
            if not predictor.load_models(args.symbol):
                print("Training initial models...")
                predictor.train_and_predict(args.symbol, args.minutes)
            
            while True:
                try:
                    result = predictor.predict_fast(args.symbol, args.minutes)
                    
                    print(f"\n[{result['prediction_time']}] {args.symbol}")
                    print(f"Current: ₹{result['current_price']:.2f} → Predicted: ₹{result['predicted_price']:.2f}")
                    print(f"Change: {result['price_change']:+.2f} ({result['price_change_pct']:+.2f}%)")
                    print(f"Confidence: {result['confidence']:.1%}")
                    
                    time.sleep(30)  # Update every 30 seconds
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(10)
            
            predictor.stop_background_updates()
            
        else:
            # Single prediction mode
            if args.retrain or not predictor.load_models(args.symbol):
                result = predictor.train_and_predict(args.symbol, args.minutes)
            else:
                result = predictor.predict_fast(args.symbol, args.minutes)
            
            # Display results
            print("\n" + "="*60)
            print("REAL-TIME STOCK PRICE PREDICTION")
            print("="*60)
            print(f"Symbol: {result['symbol']}")
            print(f"Current Price: ₹{result['current_price']:.2f}")
            print(f"Predicted Price (Next {args.minutes} mins): ₹{result['predicted_price']:.2f}")
            print(f"Expected Change: ₹{result['price_change']:.2f} ({result['price_change_pct']:+.2f}%)")
            print(f"Last Trade Time: {result['last_trade_time']}")
            print(f"Prediction Time: {result['prediction_time']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Speed: {result['prediction_speed']}")
            print(f"Total Time: {result['total_prediction_time']:.3f} seconds")
            print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
