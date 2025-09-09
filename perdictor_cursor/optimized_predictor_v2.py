#!/usr/bin/env python3
"""
Robust Stock Price Predictor - Rewritten from scratch
Designed for high-frequency intraday trading with consistent feature engineering
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from time import perf_counter
from datetime import datetime, time
from collections import deque
import threading
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional dependencies
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error


class RobustStockPredictor:
    """
    Robust stock price predictor with consistent feature engineering
    and comprehensive error handling
    """
    
    def __init__(self, db_path: str, models_dir: str = "models"):
        """
        Initialize the predictor
        
        Args:
            db_path: Path to SQLite database
            models_dir: Directory to store/load models
        """
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize caches
        self.data_cache = {}
        self.model_cache = {}
        self.feature_cache = {}
        self.scaler_cache = {}
        
        # Cache settings
        self.cache_size = 1000
        self.cache_lock = threading.Lock()
        
        # Feature engineering settings
        self.required_columns = [
            'symbol', 'ts', 'ltp', 'bid', 'ask', 'bid_qty', 'ask_qty', 
            'vwap', 'vol_traded_today', 'last_traded_qty', 'exch_feed_time',
            'bid_size', 'ask_size', 'bid_price', 'ask_price', 'tot_buy_qty',
            'tot_sell_qty', 'avg_trade_price', 'ch', 'chp', 'last_traded_time',
            'high', 'low', 'open', 'prev_close', 'high_price', 'low_price',
            'open_price', 'prev_close_price'
        ]
        
        # Define feature engineering pipeline
        self.feature_pipeline = self._define_feature_pipeline()
        
        # Setup logging
        self.setup_logging()
        
        # Optimize database
        self.optimize_database()
        
        self.logger.info("RobustStockPredictor initialized successfully")
    
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
    
    def optimize_database(self):
        """Optimize database for fast queries"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA mmap_size=268435456")
                conn.execute("PRAGMA page_size=4096")
                conn.execute("PRAGMA optimize")
                conn.execute("PRAGMA busy_timeout=30000")
                conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
                conn.execute("PRAGMA wal_autocheckpoint=1000")
                
                # Create indexes for fast queries
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts ON ticks(symbol, ts DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks(ts DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_ticks_ltp ON ticks(ltp)",
                    "CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts_ltp ON ticks(symbol, ts DESC, ltp)",
                    "CREATE INDEX IF NOT EXISTS idx_ticks_recent ON ticks(symbol, ts DESC) WHERE ts > datetime('now', '-1 day')"
                ]
                
                for index_sql in indexes:
                    conn.execute(index_sql)
                
                conn.execute("ANALYZE")
                self.logger.info("Database optimized successfully")
                
        except Exception as e:
            self.logger.error(f"Error optimizing database: {e}")
    
    def _define_feature_pipeline(self) -> Dict[str, Any]:
        """
        Define the feature engineering pipeline
        This ensures consistent features between training and prediction
        """
        return {
            'price_features': [
                'ltp', 'bid', 'ask', 'high', 'low', 'open', 'prev_close',
                'high_price', 'low_price', 'open_price', 'prev_close_price'
            ],
            'volume_features': [
                'vol_traded_today', 'last_traded_qty', 'bid_qty', 'ask_qty',
                'bid_size', 'ask_size', 'tot_buy_qty', 'tot_sell_qty'
            ],
            'derived_features': [
                'vwap', 'avg_trade_price', 'ch', 'chp'
            ],
            'technical_indicators': {
                'sma_windows': [3, 5, 10, 15, 20],
                'ema_windows': [5, 10, 15, 20],
                'rsi_windows': [7, 14],
                'bollinger_window': 20,
                'bollinger_std': 2
            }
        }
    
    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required columns exist with appropriate defaults
        This is the key to preventing feature mismatch errors
        """
        df = df.copy()
        
        # Define default values for missing columns
        defaults = {
            'bid': df['ltp'] if 'ltp' in df.columns else 0,
            'ask': df['ltp'] if 'ltp' in df.columns else 0,
            'bid_qty': 0,
            'ask_qty': 0,
            'last_traded_qty': 0,
            'bid_size': 0,
            'ask_size': 0,
            'bid_price': df['bid'] if 'bid' in df.columns else df['ltp'] if 'ltp' in df.columns else 0,
            'ask_price': df['ask'] if 'ask' in df.columns else df['ltp'] if 'ltp' in df.columns else 0,
            'tot_buy_qty': 0,
            'tot_sell_qty': 0,
            'avg_trade_price': df['ltp'] if 'ltp' in df.columns else 0,
            'ch': 0,
            'chp': 0,
            'high_price': df['high'] if 'high' in df.columns else df['ltp'] if 'ltp' in df.columns else 0,
            'low_price': df['low'] if 'low' in df.columns else df['ltp'] if 'ltp' in df.columns else 0,
            'open_price': df['open'] if 'open' in df.columns else df['ltp'] if 'ltp' in df.columns else 0,
            'prev_close_price': df['prev_close'] if 'prev_close' in df.columns else df['ltp'] if 'ltp' in df.columns else 0
        }
        
        # Add missing columns with defaults
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
                self.logger.debug(f"Added missing column {col} with default value")
        
        return df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features consistently
        This method is used for both training and prediction
        """
        df = self._ensure_required_columns(df)
        
        # Sort by timestamp
        df = df.sort_values('ts').reset_index(drop=True)
        
        # Basic price features
        df['price_change'] = df['ltp'].diff()
        df['price_change_pct'] = df['ltp'].pct_change()
        
        # Bid-ask features
        df['bid_ask_spread'] = df['ask'] - df['bid']
        df['bid_ask_spread_pct'] = df['bid_ask_spread'] / df['ltp']
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['price_vs_mid'] = df['ltp'] / df['mid_price']
        
        # Volume features
        df['volume_change'] = df['vol_traded_today'].diff()
        df['volume_change_pct'] = df['vol_traded_today'].pct_change()
        df['buy_sell_ratio'] = df['tot_buy_qty'] / (df['tot_sell_qty'] + 1e-8)
        df['buy_sell_imbalance'] = (df['tot_buy_qty'] - df['tot_sell_qty']) / (df['tot_buy_qty'] + df['tot_sell_qty'] + 1e-8)
        
        # Price position features
        df['price_vs_high'] = df['ltp'] / df['high']
        df['price_vs_low'] = df['ltp'] / df['low']
        df['price_vs_open'] = df['ltp'] / df['open']
        df['price_vs_prev_close'] = df['ltp'] / df['prev_close']
        df['high_low_range'] = (df['high'] - df['low']) / df['ltp']
        df['price_position'] = (df['ltp'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # Technical indicators
        self._add_technical_indicators(df)
        
        # Time-based features
        self._add_time_features(df)
        
        # VWAP features
        if 'vwap' in df.columns:
            df['vwap_vs_ltp'] = df['ltp'] / df['vwap']
            df['vwap_deviation'] = (df['ltp'] - df['vwap']) / df['vwap']
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame):
        """Add technical indicators"""
        pipeline = self.feature_pipeline['technical_indicators']
        
        # Moving averages
        for window in pipeline['sma_windows']:
            df[f'sma_{window}'] = df['ltp'].rolling(window=window).mean()
            df[f'price_vs_sma_{window}'] = df['ltp'] / df[f'sma_{window}']
        
        for window in pipeline['ema_windows']:
            df[f'ema_{window}'] = df['ltp'].ewm(span=window).mean()
            df[f'price_vs_ema_{window}'] = df['ltp'] / df[f'ema_{window}']
        
        # RSI
        for window in pipeline['rsi_windows']:
            delta = df['ltp'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        window = pipeline['bollinger_window']
        std = pipeline['bollinger_std']
        df[f'bb_middle_{window}'] = df['ltp'].rolling(window=window).mean()
        bb_std = df['ltp'].rolling(window=window).std()
        df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + (bb_std * std)
        df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - (bb_std * std)
        df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
        df[f'bb_position_{window}'] = (df['ltp'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'] + 1e-8)
    
    def _add_time_features(self, df: pd.DataFrame):
        """Add time-based features"""
        if 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'])
            df['hour'] = df['ts'].dt.hour
            df['minute'] = df['ts'].dt.minute
            df['time_of_day'] = df['hour'] * 60 + df['minute']
            df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
            df['is_opening_hour'] = ((df['hour'] == 9) & (df['minute'] >= 15) & (df['minute'] <= 30)).astype(int)
            df['is_closing_hour'] = ((df['hour'] == 15) & (df['minute'] >= 0) & (df['minute'] <= 30)).astype(int)
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get the list of feature columns to use for training/prediction
        This ensures consistency between training and prediction
        """
        # Exclude non-feature columns
        exclude_cols = [
            'symbol', 'ts', 'exch_feed_time', 'last_traded_time', 'target'
        ]
        
        # Get all columns except excluded ones
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Ensure we have numeric columns only
        numeric_cols = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                self.logger.warning(f"Skipping non-numeric column: {col}")
        
        return numeric_cols
    
    def fetch_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch data from database"""
        try:
            start_time = perf_counter()
            
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                query = """
                SELECT symbol, ts, ltp, bid, ask, bid_qty, ask_qty, vwap, 
                       vol_traded_today, last_traded_qty, exch_feed_time,
                       bid_size, ask_size, bid_price, ask_price, tot_buy_qty,
                       tot_sell_qty, avg_trade_price, ch, chp, last_traded_time,
                       high, low, open, prev_close, high_price, low_price,
                       open_price, prev_close_price
                FROM ticks 
                WHERE symbol = ? 
                ORDER BY ts DESC 
                LIMIT ?
                """
                
                df = pd.read_sql_query(query, conn, params=(symbol, limit))
                
                if len(df) == 0:
                    raise ValueError(f"No data found for symbol: {symbol}")
                
                fetch_time = perf_counter() - start_time
                self.logger.info(f"Fetched {len(df)} records for {symbol} in {fetch_time:.3f} seconds")
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    def get_cached_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get data from cache or fetch if not available"""
        with self.cache_lock:
            if symbol not in self.data_cache:
                # Fetch initial data
                df = self.fetch_data(symbol, limit)
                self.data_cache[symbol] = deque(df.to_dict('records'), maxlen=self.cache_size)
                self.logger.info(f"Initialized cache for {symbol} with {len(df)} records")
            
            # Convert cache to DataFrame
            cache_data = list(self.data_cache[symbol])
            df = pd.DataFrame(cache_data)
            
            if len(df) == 0:
                raise ValueError(f"No cached data available for {symbol}")
            
            return df
    
    def get_fresh_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get fresh data from database for real-time predictions"""
        try:
            start_time = perf_counter()
            
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                query = """
                SELECT symbol, ts, ltp, bid, ask, bid_qty, ask_qty, vwap, 
                       vol_traded_today, last_traded_qty, exch_feed_time,
                       bid_size, ask_size, bid_price, ask_price, tot_buy_qty,
                       tot_sell_qty, avg_trade_price, ch, chp, last_traded_time,
                       high, low, open, prev_close, high_price, low_price,
                       open_price, prev_close_price
                FROM ticks 
                WHERE symbol = ? 
                ORDER BY ts DESC 
                LIMIT ?
                """
                
                df = pd.read_sql_query(query, conn, params=(symbol, limit))
                
                if len(df) == 0:
                    raise ValueError(f"No data found for symbol: {symbol}")
                
                fetch_time = perf_counter() - start_time
                self.logger.info(f"Fetched fresh data for {symbol}: {len(df)} records in {fetch_time:.3f} seconds")
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error fetching fresh data for {symbol}: {e}")
            raise
    
    def prepare_training_data(self, df: pd.DataFrame, target_minutes: int = 15) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare training data with consistent feature engineering"""
        # Create features
        df_features = self._create_features(df)
        
        # Create target variable
        df_features['target'] = df_features['ltp'].shift(-target_minutes)
        
        # Remove rows without target
        df_clean = df_features.dropna(subset=['target'])
        
        if len(df_clean) < 50:
            raise ValueError(f"Insufficient data for training: {len(df_clean)} records")
        
        # Get feature columns
        feature_cols = self._get_feature_columns(df_clean)
        
        # Prepare X and y
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        # Remove any remaining NaN values
        X = X.fillna(0)
        
        self.logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_cols
    
    def train_models(self, symbol: str, target_minutes: int = 15) -> bool:
        """Train models for a symbol"""
        try:
            self.logger.info(f"Training models for {symbol}")
            
            # Fetch data
            df = self.fetch_data(symbol, 1000)
            
            # Prepare training data
            X, y, feature_cols = self.prepare_training_data(df, target_minutes)
            
            # Initialize models
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, max_depth=8, min_samples_split=5,
                    min_samples_leaf=2, random_state=42, n_jobs=-1
                ),
                'linear_regression': LinearRegression(),
                'ridge': Ridge(alpha=0.1)
            }
            
            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                models['xgboost'] = xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )
            
            # Add LightGBM if available
            if LIGHTGBM_AVAILABLE:
                models['lightgbm'] = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    verbose=-1
                )
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            model_scores = {}
            
            for name, model in models.items():
                scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    score = mean_absolute_error(y_val, y_pred)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                model_scores[name] = avg_score
                self.logger.info(f"Trained {name} model with MAE: {avg_score:.4f}")
            
            # Train final models on full dataset
            for name, model in models.items():
                model.fit(X_scaled, y)
            
            # Save models and metadata
            model_dir = self.models_dir / symbol.replace(':', '_')
            model_dir.mkdir(exist_ok=True)
            
            # Save models
            for name, model in models.items():
                joblib.dump(model, model_dir / f"{name}_model.pkl")
            
            # Save scaler and feature columns
            joblib.dump(scaler, model_dir / "scaler.pkl")
            joblib.dump(feature_cols, model_dir / "feature_columns.pkl")
            joblib.dump(model_scores, model_dir / "model_scores.pkl")
            
            # Cache models
            self.model_cache[symbol] = models
            self.scaler_cache[symbol] = scaler
            self.feature_cache[symbol] = feature_cols
            
            self.logger.info(f"Successfully trained models for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models for {symbol}: {e}")
            return False
    
    def load_models(self, symbol: str) -> bool:
        """Load models for a symbol"""
        try:
            model_dir = self.models_dir / symbol.replace(':', '_')
            
            if not model_dir.exists():
                self.logger.warning(f"No models found for {symbol}, training new models...")
                return self.train_models(symbol)
            
            # Load models
            models = {}
            model_files = {
                'random_forest': 'random_forest_model.pkl',
                'linear_regression': 'linear_regression_model.pkl',
                'ridge': 'ridge_model.pkl'
            }
            
            if XGBOOST_AVAILABLE:
                model_files['xgboost'] = 'xgboost_model.pkl'
            if LIGHTGBM_AVAILABLE:
                model_files['lightgbm'] = 'lightgbm_model.pkl'
            
            for name, filename in model_files.items():
                model_path = model_dir / filename
                if model_path.exists():
                    models[name] = joblib.load(model_path)
                else:
                    self.logger.warning(f"Model {name} not found for {symbol}")
            
            if not models:
                raise ValueError(f"No models found for {symbol}")
            
            # Load scaler and feature columns
            scaler = joblib.load(model_dir / "scaler.pkl")
            feature_cols = joblib.load(model_dir / "feature_columns.pkl")
            
            # Cache loaded models
            self.model_cache[symbol] = models
            self.scaler_cache[symbol] = scaler
            self.feature_cache[symbol] = feature_cols
            
            self.logger.info(f"Loaded {len(models)} models for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models for {symbol}: {e}")
            return False
    
    def predict(self, symbol: str, target_minutes: int = 15) -> Dict[str, Any]:
        """Make prediction for a symbol"""
        try:
            if symbol not in self.model_cache:
                if not self.load_models(symbol):
                    raise ValueError(f"Could not load models for {symbol}")
            
            # Get fresh data from database for real-time predictions
            df = self.get_fresh_data(symbol, 1000)
            
            # Create features
            df_features = self._create_features(df)
            
            # Get latest data point
            latest_data = df_features.iloc[-1:].copy()
            
            # Get feature columns used in training
            feature_cols = self.feature_cache[symbol]
            
            # Prepare features for prediction
            X_pred = latest_data[feature_cols].fillna(0)
            
            # Scale features
            scaler = self.scaler_cache[symbol]
            X_pred_scaled = scaler.transform(X_pred)
            
            # Make predictions
            models = self.model_cache[symbol]
            predictions = {}
            
            for name, model in models.items():
                try:
                    pred = model.predict(X_pred_scaled)[0]
                    predictions[name] = pred
                except Exception as e:
                    self.logger.error(f"Error predicting with {name}: {e}")
                    predictions[name] = None
            
            # Calculate ensemble prediction
            valid_predictions = {k: v for k, v in predictions.items() if v is not None}
            if not valid_predictions:
                raise ValueError("No valid predictions available")
            
            # Weighted ensemble
            model_weights = {
                'xgboost': 0.4, 'lightgbm': 0.3, 'random_forest': 0.2,
                'linear_regression': 0.05, 'ridge': 0.05
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
            
            # Apply realistic bounds
            current_price = latest_data['ltp'].iloc[0]
            ensemble_pred = self._apply_realistic_bounds(ensemble_pred, current_price, target_minutes)
            
            # Calculate confidence
            confidence = self._calculate_confidence(valid_predictions, ensemble_pred)
            
            # Prepare result
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': ensemble_pred,
                'confidence': confidence,
                'price_change': ensemble_pred - current_price,
                'price_change_pct': ((ensemble_pred - current_price) / current_price) * 100,
                'target_minutes': target_minutes,
                'timestamp': datetime.now().isoformat(),
                'last_trade_time': latest_data['last_traded_time'].iloc[0] if 'last_traded_time' in latest_data.columns else None
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {e}")
            raise
    
    def _apply_realistic_bounds(self, predicted_price: float, current_price: float, target_minutes: int) -> float:
        """Apply realistic bounds to predictions"""
        # Conservative bounds for intraday trading
        max_change_pct = min(2.0, target_minutes * 0.1)  # Max 2% change for 15-min prediction
        max_change = current_price * (max_change_pct / 100)
        
        min_price = current_price - max_change
        max_price = current_price + max_change
        
        bounded_price = max(min_price, min(max_price, predicted_price))
        
        if bounded_price != predicted_price:
            self.logger.debug(f"Applied realistic bounds: {predicted_price:.2f} -> {bounded_price:.2f}")
        
        return bounded_price
    
    def _calculate_confidence(self, predictions: Dict[str, float], ensemble_pred: float) -> float:
        """Calculate prediction confidence"""
        if len(predictions) < 2:
            return 0.3
        
        # Calculate coefficient of variation
        pred_values = list(predictions.values())
        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)
        
        if mean_pred == 0:
            return 0.3
        
        cv = std_pred / abs(mean_pred)
        
        # Convert CV to confidence (lower CV = higher confidence)
        confidence = max(0.1, min(0.8, 0.8 - cv))
        
        return confidence
    
    def generate_trading_signal(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal based on prediction"""
        current_price = prediction_result['current_price']
        predicted_price = prediction_result['predicted_price']
        confidence = prediction_result['confidence']
        
        # Calculate potential profit
        potential_profit = abs(predicted_price - current_price)
        potential_profit_pct = (potential_profit / current_price) * 100 if current_price > 0 else 0
        
        # Signal criteria
        min_confidence = 0.7  # 70% minimum confidence
        min_profit_pct = 0.5  # 0.5% minimum profit
        
        # Determine signal
        if confidence >= min_confidence and potential_profit_pct >= min_profit_pct:
            if predicted_price > current_price:
                signal = "BUY"
                action = "LONG"
            else:
                signal = "SELL"
                action = "SHORT"
        else:
            signal = "HOLD"
            action = "WAIT"
        
        # Risk assessment
        if potential_profit_pct > 2.0:
            risk_level = "HIGH"
        elif potential_profit_pct > 1.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'signal': signal,
            'action': action,
            'confidence': confidence,
            'potential_profit': potential_profit,
            'potential_profit_pct': potential_profit_pct,
            'risk_level': risk_level,
            'current_price': current_price,
            'predicted_price': predicted_price
        }
    
    def is_market_hours(self) -> bool:
        """Check if current time is market hours"""
        now = datetime.now().time()
        market_open = time(9, 15)  # 9:15 AM
        market_close = time(15, 30)  # 3:30 PM
        return market_open <= now <= market_close


def main():
    """Main function for testing"""
    predictor = RobustStockPredictor('/home/karthik/market.db')
    
    # Test with a symbol
    symbol = 'NSE:RELIANCE-EQ'
    
    try:
        # Load or train models
        if not predictor.load_models(symbol):
            print(f"Failed to load/train models for {symbol}")
            return
        
        # Make prediction
        result = predictor.predict(symbol, 15)
        signal = predictor.generate_trading_signal(result)
        
        print(f"\n🎯 PREDICTION RESULTS for {symbol}")
        print("=" * 50)
        print(f"Current Price: ₹{result['current_price']:.2f}")
        print(f"Predicted Price: ₹{result['predicted_price']:.2f}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Price Change: ₹{result['price_change']:.2f} ({result['price_change_pct']:.2f}%)")
        
        print(f"\n📊 TRADING SIGNAL")
        print("=" * 30)
        print(f"Signal: {signal['signal']}")
        print(f"Action: {signal['action']}")
        print(f"Potential Profit: ₹{signal['potential_profit']:.2f} ({signal['potential_profit_pct']:.2f}%)")
        print(f"Risk Level: {signal['risk_level']}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
