#!/usr/bin/env python3
"""
Optimized Real-Time Stock Predictor with Data Caching
Minimizes database queries and maximizes prediction speed
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
from collections import deque
import pickle
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

warnings.filterwarnings('ignore')

class OptimizedStockPredictor:
    """
    Optimized real-time stock predictor with:
    - Data caching and streaming
    - Minimal database queries
    - In-memory data management
    - Real-time updates
    """
    
    def __init__(self, db_path: str = "/home/karthik/market.db", model_dir: str = "optimized_models"):
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Data caching for real-time trading
        self.data_cache = {}  # symbol -> deque of recent data
        self.cache_size = 2000  # Keep last 2000 records per symbol for better accuracy
        self.last_fetch_time = {}  # symbol -> last fetch timestamp
        self.fetch_interval = 5  # Fetch new data every 5 seconds during market hours
        self.last_prediction_time = {}  # symbol -> last prediction timestamp
        
        # Market hours configuration (IST)
        self.market_open_hour = 9
        self.market_open_minute = 15
        self.market_close_hour = 15
        self.market_close_minute = 30
        
        # Trading session tracking
        self.is_market_open = False
        self.session_start_time = None
        
        # Model cache
        self.model_cache = {}  # symbol -> trained models
        self.feature_cache = {}  # symbol -> feature columns
        
        # Optimized models for intraday trading accuracy
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,  # Increased for better accuracy
                max_depth=8,       # Deeper trees for complex patterns
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=0.1)  # Reduced regularization
        }
        
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,  # Increased for better accuracy
                max_depth=6,       # Deeper for complex patterns
                learning_rate=0.05,  # Lower learning rate for stability
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Background threads
        self.data_thread = None
        self.model_thread = None
        self.stop_threads = False
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('optimized_predictor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours (9:15 AM - 3:30 PM IST)"""
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_start = (self.market_open_hour, self.market_open_minute)
        market_end = (self.market_close_hour, self.market_close_minute)
        
        current_time = (current_hour, current_minute)
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        is_weekday = now.weekday() < 5
        
        if not is_weekday:
            return False
            
        return market_start <= current_time <= market_end
    
    def get_market_status(self) -> Dict:
        """Get current market status and time to open/close"""
        now = datetime.now()
        is_open = self.is_market_hours()
        
        if is_open:
            # Calculate time to close
            close_time = now.replace(hour=self.market_close_hour, minute=self.market_close_minute, second=0, microsecond=0)
            time_to_close = close_time - now
            return {
                'is_open': True,
                'status': 'OPEN',
                'time_to_close': str(time_to_close).split('.')[0],
                'current_time': now.strftime('%H:%M:%S')
            }
        else:
            # Calculate time to open (next trading day)
            if now.hour < self.market_open_hour or (now.hour == self.market_open_hour and now.minute < self.market_open_minute):
                # Market opens today
                open_time = now.replace(hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0)
            else:
                # Market opens next trading day
                days_ahead = 1
                if now.weekday() == 4:  # Friday
                    days_ahead = 3  # Monday
                elif now.weekday() == 5:  # Saturday
                    days_ahead = 2  # Monday
                
                open_time = (now + timedelta(days=days_ahead)).replace(
                    hour=self.market_open_hour, 
                    minute=self.market_open_minute, 
                    second=0, 
                    microsecond=0
                )
            
            time_to_open = open_time - now
            return {
                'is_open': False,
                'status': 'CLOSED',
                'time_to_open': str(time_to_open).split('.')[0],
                'current_time': now.strftime('%H:%M:%S')
            }
    
    def connect_db(self) -> sqlite3.Connection:
        """Create database connection with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}")
            raise
    
    def fetch_initial_data(self, symbol: str, limit: int = 2000) -> pd.DataFrame:
        """Fetch initial data for training (one-time)"""
        start_time = perf_counter()
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
            
            fetch_time = perf_counter() - start_time
            self.logger.info(f"Fetched {len(df)} initial records for {symbol} in {fetch_time:.3f} seconds")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching initial data for {symbol}: {e}")
            raise
    
    def fetch_new_data(self, symbol: str, since_timestamp: int) -> pd.DataFrame:
        """Fetch only new data since last update - optimized for real-time trading"""
        start_time = perf_counter()
        
        # Adjust limit based on market hours
        limit = 200 if self.is_market_hours() else 100
        
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
        WHERE symbol = ? AND ts > ?
        ORDER BY ts ASC
        LIMIT ?
        """
        
        try:
            with self.connect_db() as conn:
                df = pd.read_sql_query(query, conn, params=(symbol, since_timestamp, limit))
                
            if not df.empty:
                # Convert timestamp columns
                df['ts'] = pd.to_datetime(df['ts'], unit='s')
                df['exch_feed_time'] = pd.to_datetime(df['exch_feed_time'], unit='s')
                df['last_traded_time'] = pd.to_datetime(df['last_traded_time'], unit='s')
                
                fetch_time = perf_counter() - start_time
                self.logger.info(f"Fetched {len(df)} new records for {symbol} in {fetch_time:.3f} seconds")
            else:
                fetch_time = perf_counter() - start_time
                self.logger.info(f"No new data for {symbol} (checked in {fetch_time:.3f} seconds)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching new data for {symbol}: {e}")
            return pd.DataFrame()
    
    
    def initialize_data_cache(self, symbol: str):
        """Initialize data cache for a symbol"""
        if symbol not in self.data_cache:
            # Fetch initial data
            df = self.fetch_initial_data(symbol, 1000)
            
            # Convert to deque for efficient appending
            data_list = df.to_dict('records')
            self.data_cache[symbol] = deque(data_list, maxlen=self.cache_size)
            
            # Record last fetch time
            if not data_list:
                self.last_fetch_time[symbol] = 0
            else:
                self.last_fetch_time[symbol] = int(data_list[-1]['ts'].timestamp())
            
            self.logger.info(f"Initialized data cache for {symbol} with {len(data_list)} records")
    
    def update_data_cache(self, symbol: str):
        """Update data cache with new data"""
        if symbol not in self.data_cache:
            self.initialize_data_cache(symbol)
            return
        
        # Get last timestamp
        last_timestamp = self.last_fetch_time.get(symbol, 0)
        
        # Fetch new data
        new_data = self.fetch_new_data(symbol, last_timestamp)
        
        if not new_data.empty:
            # Add new data to cache
            new_records = new_data.to_dict('records')
            for record in new_records:
                self.data_cache[symbol].append(record)
            
            # Update last fetch time
            self.last_fetch_time[symbol] = int(new_records[-1]['ts'].timestamp())
            
            self.logger.info(f"Updated cache for {symbol} with {len(new_records)} new records")
    
    def get_cached_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get data from cache (no database query)"""
        start_time = perf_counter()
        
        if symbol not in self.data_cache:
            self.initialize_data_cache(symbol)
        
        # Convert deque to DataFrame
        data_list = list(self.data_cache[symbol])
        if not data_list:
            raise ValueError(f"No cached data for {symbol}")
        
        df = pd.DataFrame(data_list)
        
        # Take last 'limit' records
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        
        cache_time = perf_counter() - start_time
        self.logger.info(f"Retrieved {len(df)} records from cache for {symbol} in {cache_time:.3f} seconds")
        
        return df
    
    def create_fast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features for intraday trading accuracy"""
        df = df.copy()
        
        # Essential price features
        df['price_change'] = df['ltp'].diff()
        df['price_change_pct'] = df['ltp'].pct_change()
        
        # Bid-ask features (handle missing values consistently)
        # Ensure ask and bid columns exist with default values
        if 'ask' not in df.columns:
            df['ask'] = df['ltp']  # Use ltp as default for ask
        if 'bid' not in df.columns:
            df['bid'] = df['ltp']  # Use ltp as default for bid
        if 'last_traded_qty' not in df.columns:
            df['last_traded_qty'] = 0  # Default value for missing column
        if 'ch' not in df.columns:
            df['ch'] = 0  # Default value for missing column
        if 'prev_close_price' not in df.columns:
            df['prev_close_price'] = df['prev_close']  # Use prev_close as default
        if 'high_price' not in df.columns:
            df['high_price'] = df['high']  # Use high as default
        if 'low_price' not in df.columns:
            df['low_price'] = df['low']  # Use low as default
        if 'open_price' not in df.columns:
            df['open_price'] = df['open']  # Use open as default
        
        df['bid_ask_spread'] = (df['ask'] - df['bid']).fillna(0)
        df['bid_ask_spread_pct'] = ((df['ask'] - df['bid']) / df['ltp']).fillna(0)
        
        # Enhanced volume features for intraday
        df['volume_change'] = df['vol_traded_today'].diff()
        df['volume_change_pct'] = df['vol_traded_today'].pct_change()
        df['buy_sell_ratio'] = df['tot_buy_qty'] / (df['tot_sell_qty'] + 1e-8)
        df['buy_sell_imbalance'] = (df['tot_buy_qty'] - df['tot_sell_qty']) / (df['tot_buy_qty'] + df['tot_sell_qty'] + 1e-8)
        
        # Price momentum features
        df['price_vs_high'] = df['ltp'] / df['high']
        df['price_vs_low'] = df['ltp'] / df['low']
        df['price_vs_open'] = df['ltp'] / df['open']
        df['price_vs_prev_close'] = df['ltp'] / df['prev_close']
        df['high_low_range'] = (df['high'] - df['low']) / df['ltp']
        df['price_position'] = (df['ltp'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # Multiple timeframe moving averages
        for window in [3, 5, 10, 12, 15, 20, 26]:
            df[f'sma_{window}'] = df['ltp'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['ltp'].ewm(span=window).mean()
            df[f'price_vs_sma_{window}'] = df['ltp'] / df[f'sma_{window}']
        
        # Enhanced RSI with multiple timeframes
        for window in [7, 14, 21]:
            delta = df['ltp'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # MACD-like indicators (after ema_12 and ema_26 are created)
        df['macd_12_26'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd_12_26'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd_12_26'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['ltp'].rolling(window=20).mean()
        df['bb_std'] = df['ltp'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['ltp'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # Volume-based indicators
        df['volume_sma_10'] = df['vol_traded_today'].rolling(window=10).mean()
        df['volume_ratio'] = df['vol_traded_today'] / df['volume_sma_10']
        df['price_volume_trend'] = df['price_change_pct'] * df['volume_ratio']
        
        # Lag features for multiple timeframes
        for lag in [1, 2, 3, 5, 10]:
            df[f'ltp_lag_{lag}'] = df['ltp'].shift(lag)
            df[f'volume_lag_{lag}'] = df['vol_traded_today'].shift(lag)
            df[f'price_change_lag_{lag}'] = df['price_change_pct'].shift(lag)
        
        # Rolling statistics for volatility
        for window in [5, 10, 20]:
            df[f'ltp_std_{window}'] = df['ltp'].rolling(window=window).std()
            df[f'ltp_mean_{window}'] = df['ltp'].rolling(window=window).mean()
            df[f'volatility_{window}'] = df[f'ltp_std_{window}'] / df[f'ltp_mean_{window}']
        
        # Time-based features for intraday patterns
        df['hour'] = df['ts'].dt.hour
        df['minute'] = df['ts'].dt.minute
        df['time_of_day'] = df['hour'] * 60 + df['minute']
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        df['is_opening_hour'] = ((df['hour'] == 9) & (df['minute'] >= 15) & (df['minute'] <= 30)).astype(int)
        df['is_closing_hour'] = ((df['hour'] == 15) & (df['minute'] >= 0) & (df['minute'] <= 30)).astype(int)
        
        # VWAP features
        df['vwap_vs_ltp'] = df['ltp'] / df['vwap']
        df['vwap_deviation'] = (df['ltp'] - df['vwap']) / df['vwap']
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, target_minutes: int = 15) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with target variable"""
        df['target'] = df['ltp'].shift(-target_minutes)
        df_clean = df.dropna(subset=['target'])
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        if len(df_clean) < 50:
            raise ValueError(f"Insufficient data for training: {len(df_clean)} records")
        
        exclude_cols = ['symbol', 'ts', 'exch_feed_time', 'last_traded_time', 'target', 'ltp']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        # Ensure all essential features are included
        essential_features = ['prev_close', 'high', 'low', 'open', 'vol_traded_today', 'price_change_pct', 'bid_ask_spread', 'bid_ask_spread_pct', 'ask', 'bid', 'last_traded_qty', 'ch', 'prev_close_price', 'high_price', 'low_price', 'open_price']
        
        # Use all available features for consistency with prediction
        valid_feature_cols = feature_cols.copy()
        
        # Ensure all essential features are included
        for feature in essential_features:
            if feature in df_clean.columns and feature not in valid_feature_cols:
                valid_feature_cols.append(feature)
                self.logger.warning(f"Added missing essential feature: {feature}")
        
        # Fill any remaining NaN values to ensure consistency
        for col in valid_feature_cols:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(0)
        
        X = df_clean[valid_feature_cols]
        y = df_clean['target']
        
        return X, y, valid_feature_cols
    
    def train_models_fast(self, X: pd.DataFrame, y: pd.Series, symbol: str, feature_cols: List[str]):
        """Fast model training"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        scores = {}
        trained_models = {}
        
        for name, model in self.models.items():
            try:
                # Quick training
                model.fit(X_scaled, y)
                trained_models[name] = model
                
                # Simple validation score
                pred = model.predict(X_scaled)
                mae = mean_absolute_error(y, pred)
                scores[name] = mae
                
                self.logger.info(f"Trained {name} model with MAE: {mae:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name} model: {e}")
                scores[name] = float('inf')
        
        # Cache models and features
        self.model_cache[symbol] = trained_models
        self.feature_cache[symbol] = feature_cols
        
        # Save to disk
        symbol_clean = symbol.replace(':', '_').replace('-', '_')
        symbol_model_dir = self.model_dir / symbol_clean
        symbol_model_dir.mkdir(exist_ok=True)
        
        for name, model in trained_models.items():
            joblib.dump(model, symbol_model_dir / f"{name}_model.pkl")
        joblib.dump(self.scaler, symbol_model_dir / "scaler.pkl")
        joblib.dump(feature_cols, symbol_model_dir / "feature_columns.pkl")
        
        self.is_trained = True
        return scores
    
    def load_models(self, symbol: str) -> bool:
        """Load pre-trained models from cache or disk"""
        # Check cache first
        if symbol in self.model_cache and symbol in self.feature_cache:
            self.logger.info(f"Loaded models for {symbol} from cache")
            return True
        
        # Load from disk
        try:
            symbol_clean = symbol.replace(':', '_').replace('-', '_')
            model_dir = self.model_dir / symbol_clean
            
            if not model_dir.exists():
                return False
            
            trained_models = {}
            for name in self.models.keys():
                model_path = model_dir / f"{name}_model.pkl"
                if model_path.exists():
                    trained_models[name] = joblib.load(model_path)
                else:
                    return False
            
            self.scaler = joblib.load(model_dir / "scaler.pkl")
            feature_cols = joblib.load(model_dir / "feature_columns.pkl")
            
            # Cache loaded models
            self.model_cache[symbol] = trained_models
            self.feature_cache[symbol] = feature_cols
            
            self.logger.info(f"Loaded models for {symbol} from disk")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def predict_fast(self, symbol: str, target_minutes: int = 15) -> Dict:
        """Ultra-fast prediction using cached data"""
        if symbol not in self.model_cache or symbol not in self.feature_cache:
            raise ValueError(f"Models not loaded for {symbol}")
        
        try:
            prediction_start = perf_counter()
            
            # Get data from cache (no database query!)
            df = self.get_cached_data(symbol, 1000)
            
            # Create features
            try:
                df_features = self.create_fast_features(df)
                if df_features is None or len(df_features) == 0:
                    raise ValueError(f"No features created for {symbol}")
            except Exception as e:
                self.logger.error(f"Error creating features for {symbol}: {e}")
                raise ValueError(f"Feature creation failed for {symbol}: {e}")
            
            # Get latest data point
            latest_data = df_features.iloc[-1:].copy()
            
            # Prepare features for prediction
            feature_cols = self.feature_cache[symbol]
            
            # Create X_pred with all available features first
            X_pred = latest_data.copy()
            
            # Add missing features with default values
            for col in feature_cols:
                if col not in X_pred.columns:
                    X_pred[col] = 0.0
                    self.logger.warning(f"Added missing feature during prediction: {col}")
            
            # Select only the features that were used in training
            X_pred = X_pred[feature_cols].fillna(0)
            
            # Ensure features are in the same order as during training
            # Get the feature names from the scaler
            scaler_feature_names = self.scaler.feature_names_in_
            
            # Add missing features with default values
            for feature_name in scaler_feature_names:
                if feature_name not in X_pred.columns:
                    X_pred[feature_name] = 0.0
                    self.logger.warning(f"Added missing feature during prediction: {feature_name}")
            
            # Remove extra features that weren't in training
            X_pred_ordered = X_pred[scaler_feature_names]
            
            # Ensure we have exactly the right number of features
            if len(X_pred_ordered.columns) != len(scaler_feature_names):
                self.logger.error(f"Feature count mismatch: expected {len(scaler_feature_names)}, got {len(X_pred_ordered.columns)}")
                # Create a DataFrame with exactly the right features
                X_pred_final = pd.DataFrame(index=X_pred_ordered.index)
                for feature_name in scaler_feature_names:
                    if feature_name in X_pred_ordered.columns:
                        X_pred_final[feature_name] = X_pred_ordered[feature_name]
                    else:
                        X_pred_final[feature_name] = 0.0
                X_pred_ordered = X_pred_final
            
            X_pred_scaled = self.scaler.transform(X_pred_ordered)
            
            # Make predictions with cached models
            predictions = {}
            trained_models = self.model_cache[symbol]
            
            # Get current price for bounds calculation
            current_price = latest_data['ltp'].iloc[0]
            
            for name, model in trained_models.items():
                try:
                    pred = model.predict(X_pred_scaled)[0]
                    # Apply realistic bounds to individual predictions
                    pred_bounded = self._apply_realistic_bounds(pred, current_price, target_minutes)
                    predictions[name] = pred_bounded
                except Exception as e:
                    self.logger.error(f"Error predicting with {name}: {e}")
                    predictions[name] = None
            
            # Calculate ensemble prediction
            valid_predictions = {k: v for k, v in predictions.items() if v is not None}
            if valid_predictions:
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
            
            # Get last trade time
            last_trade_time = latest_data['last_traded_time'].iloc[0]
            
            # Ensemble prediction is already bounded from individual predictions
            
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
                'prediction_speed': 'ultra_fast',
                'data_source': 'cache',
                'total_prediction_time': total_time
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {e}")
            raise
    
    def _apply_realistic_bounds(self, predicted_price: float, current_price: float, target_minutes: int) -> float:
        """Apply realistic bounds to predictions for intraday trading"""
        # More conservative bounds for all predictions
        if self.is_market_hours():
            # During market hours: allow up to 2% change for 15-min predictions
            max_change_pct = min(2.0, target_minutes * 0.1)
        else:
            # Outside market hours: very conservative bounds
            max_change_pct = min(0.5, target_minutes * 0.03)
        
        max_change = current_price * (max_change_pct / 100)
        
        min_price = current_price - max_change
        max_price = current_price + max_change
        
        bounded_price = max(min_price, min(max_price, predicted_price))
        
        if bounded_price != predicted_price:
            self.logger.info(f"Applied realistic bounds: {predicted_price:.2f} -> {bounded_price:.2f} (market hours: {self.is_market_hours()})")
        
        return bounded_price
    
    def _calculate_confidence(self, predictions: Dict, current_price: float) -> float:
        """Calculate prediction confidence based on model agreement"""
        valid_preds = [v for v in predictions.values() if v is not None]
        if len(valid_preds) < 2:
            return 0.3  # Lower default confidence
        
        mean_pred = np.mean(valid_preds)
        std_pred = np.std(valid_preds)
        
        # More realistic confidence calculation
        if mean_pred == 0:
            return 0.3
            
        # Calculate coefficient of variation
        cv = std_pred / abs(mean_pred)
        
        # More conservative confidence calculation
        # Higher CV (more disagreement) = lower confidence
        confidence = max(0.1, min(0.8, 0.8 - cv))  # Cap at 80% max confidence
        
        return confidence
    
    def generate_trading_signal(self, result: Dict) -> Dict:
        """Generate trading signals for profit-making"""
        current_price = result['current_price']
        predicted_price = result['predicted_price']
        confidence = result['confidence']
        price_change_pct = result['price_change_pct']
        
        # Calculate potential profit/loss
        potential_profit = abs(predicted_price - current_price)
        potential_profit_pct = (potential_profit / current_price) * 100 if current_price > 0 else 0
        
        # Risk management parameters
        min_confidence = 0.6  # Minimum confidence for trading
        min_profit_pct = 0.5  # Minimum 0.5% profit potential
        max_risk_pct = 2.0    # Maximum 2% risk
        
        # Generate signal
        signal = "HOLD"
        action = "WAIT"
        risk_level = "LOW"
        
        if confidence >= min_confidence and potential_profit_pct >= min_profit_pct:
            # Use actual price change for signal direction
            actual_price_change_pct = ((predicted_price - current_price) / current_price) * 100
            if actual_price_change_pct > 0:
                signal = "BUY"
                action = "LONG"
                risk_level = "MEDIUM" if potential_profit_pct < 1.0 else "HIGH"
            else:
                signal = "SELL"
                action = "SHORT"
                risk_level = "MEDIUM" if potential_profit_pct < 1.0 else "HIGH"
        elif potential_profit_pct > max_risk_pct:
            signal = "AVOID"
            action = "HIGH_RISK"
            risk_level = "VERY_HIGH"
        
        # Calculate stop loss and take profit levels
        if signal in ["BUY", "SELL"]:
            stop_loss_pct = min(1.0, potential_profit_pct * 0.5)  # 50% of expected profit
            take_profit_pct = potential_profit_pct * 0.8  # 80% of expected profit
            
            if signal == "BUY":
                stop_loss = current_price * (1 - stop_loss_pct / 100)
                take_profit = current_price * (1 + take_profit_pct / 100)
            else:  # SELL
                stop_loss = current_price * (1 + stop_loss_pct / 100)
                take_profit = current_price * (1 - take_profit_pct / 100)
        else:
            stop_loss = None
            take_profit = None
        
        return {
            'signal': signal,
            'action': action,
            'risk_level': risk_level,
            'potential_profit': potential_profit,
            'potential_profit_pct': potential_profit_pct,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence_threshold_met': confidence >= min_confidence,
            'profit_threshold_met': potential_profit_pct >= min_profit_pct,
            'recommendation': f"{signal} - {action} - Risk: {risk_level}"
        }
    
    def start_background_threads(self):
        """Start background threads for data and model updates"""
        if self.data_thread is None or not self.data_thread.is_alive():
            self.stop_threads = False
            self.data_thread = threading.Thread(target=self._data_update_loop)
            self.data_thread.daemon = True
            self.data_thread.start()
            self.logger.info("Started data update thread")
        
        if self.model_thread is None or not self.model_thread.is_alive():
            self.model_thread = threading.Thread(target=self._model_update_loop)
            self.model_thread.daemon = True
            self.model_thread.start()
            self.logger.info("Started model update thread")
    
    def stop_background_threads(self):
        """Stop background threads"""
        self.stop_threads = True
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join()
        if self.model_thread and self.model_thread.is_alive():
            self.model_thread.join()
        self.logger.info("Stopped background threads")
    
    def _data_update_loop(self):
        """Background loop for data updates"""
        while not self.stop_threads:
            try:
                for symbol in self.data_cache.keys():
                    self.update_data_cache(symbol)
                time.sleep(self.fetch_interval)
            except Exception as e:
                self.logger.error(f"Error in data update loop: {e}")
                time.sleep(60)
    
    def _model_update_loop(self):
        """Background loop for model updates - DISABLED to prevent feature mismatch"""
        while not self.stop_threads:
            try:
                # Disabled automatic model retraining to prevent feature mismatch issues
                # Models are now only retrained manually when needed
                self.logger.debug("Model update loop running (retraining disabled)")
                time.sleep(300)  # 5 minutes
            except Exception as e:
                self.logger.error(f"Error in model update loop: {e}")
                time.sleep(60)
    
    def train_and_predict(self, symbol: str, target_minutes: int = 15) -> Dict:
        """Train models and make prediction for a symbol"""
        try:
            self.logger.info(f"Starting optimized training and prediction for {symbol}")
            
            # Initialize data cache
            self.initialize_data_cache(symbol)
            
            # Get data from cache
            df = self.get_cached_data(symbol, 2000)
            
            # Create features
            df_features = self.create_fast_features(df)
            
            # Prepare training data
            X, y, feature_cols = self.prepare_training_data(df_features, target_minutes)
            
            # Train models
            scores = self.train_models_fast(X, y, symbol, feature_cols)
            self.logger.info(f"Optimized training completed. Scores: {scores}")
            
            # Make prediction
            result = self.predict_fast(symbol, target_minutes)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in train_and_predict for {symbol}: {e}")
            raise


def main():
    """Main function for optimized predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Real-Time Stock Price Predictor')
    parser.add_argument('symbol', help='Stock symbol to predict')
    parser.add_argument('--minutes', type=int, default=15, help='Minutes ahead to predict')
    parser.add_argument('--retrain', action='store_true', help='Force retrain models')
    parser.add_argument('--continuous', action='store_true', help='Run continuous predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = OptimizedStockPredictor()
    
    try:
        if args.continuous:
            # Continuous mode
            print(f"Starting optimized continuous predictions for {args.symbol}")
            predictor.start_background_threads()
            
            if not predictor.load_models(args.symbol):
                print("Training initial models...")
                predictor.train_and_predict(args.symbol, args.minutes)
            
            while True:
                try:
                    result = predictor.predict_fast(args.symbol, args.minutes)
                    trading_signal = predictor.generate_trading_signal(result)
                    market_status = predictor.get_market_status()
                    
                    print(f"\n[{result['prediction_time']}] {args.symbol}")
                    print(f"💰 Current: ₹{result['current_price']:.2f} → 🎯 Predicted: ₹{result['predicted_price']:.2f}")
                    print(f"📈 Change: {result['price_change']:+.2f} ({result['price_change_pct']:+.2f}%)")
                    print(f"🎯 Confidence: {result['confidence']:.1%}")
                    print(f"📊 Signal: {trading_signal['signal']} | Action: {trading_signal['action']} | Risk: {trading_signal['risk_level']}")
                    print(f"💵 Profit Potential: ₹{trading_signal['potential_profit']:.2f} ({trading_signal['potential_profit_pct']:.2f}%)")
                    print(f"⚡ Speed: {result['prediction_speed']} | Data: {result['data_source']} | Time: {result['total_prediction_time']:.3f}s")
                    print(f"🏪 Market: {market_status['status']} | Time: {market_status['current_time']}")
                    
                    time.sleep(5)  # Update every 5 seconds during market hours
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(10)
            
            predictor.stop_background_threads()
            
        else:
            # Single prediction mode
            if args.retrain or not predictor.load_models(args.symbol):
                result = predictor.train_and_predict(args.symbol, args.minutes)
            else:
                result = predictor.predict_fast(args.symbol, args.minutes)
            
            # Generate trading signal
            trading_signal = predictor.generate_trading_signal(result)
            market_status = predictor.get_market_status()
            
            # Display results
            print("\n" + "="*80)
            print("🚀 INTRADAY TRADING PREDICTION SYSTEM 🚀")
            print("="*80)
            print(f"📊 Symbol: {result['symbol']}")
            print(f"💰 Current Price: ₹{result['current_price']:.2f}")
            print(f"🎯 Predicted Price (Next {args.minutes} mins): ₹{result['predicted_price']:.2f}")
            print(f"📈 Expected Change: ₹{result['price_change']:.2f} ({result['price_change_pct']:+.2f}%)")
            print(f"⏰ Last Trade Time: {result['last_trade_time']}")
            print(f"🕐 Prediction Time: {result['prediction_time']}")
            print(f"🎯 Confidence: {result['confidence']:.2%}")
            print(f"⚡ Speed: {result['prediction_speed']}")
            print(f"📡 Data Source: {result['data_source']}")
            print(f"⏱️  Total Time: {result['total_prediction_time']:.3f} seconds")
            print("-" * 80)
            print("📋 TRADING SIGNALS:")
            print(f"🎯 Signal: {trading_signal['signal']}")
            print(f"📊 Action: {trading_signal['action']}")
            print(f"⚠️  Risk Level: {trading_signal['risk_level']}")
            print(f"💵 Potential Profit: ₹{trading_signal['potential_profit']:.2f} ({trading_signal['potential_profit_pct']:.2f}%)")
            if trading_signal['stop_loss']:
                print(f"🛑 Stop Loss: ₹{trading_signal['stop_loss']:.2f}")
                print(f"🎯 Take Profit: ₹{trading_signal['take_profit']:.2f}")
            print(f"💡 Recommendation: {trading_signal['recommendation']}")
            print("-" * 80)
            print("🏪 MARKET STATUS:")
            print(f"📈 Status: {market_status['status']}")
            print(f"🕐 Current Time: {market_status['current_time']}")
            if market_status['is_open']:
                print(f"⏰ Time to Close: {market_status['time_to_close']}")
            else:
                print(f"⏰ Time to Open: {market_status['time_to_open']}")
            print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
