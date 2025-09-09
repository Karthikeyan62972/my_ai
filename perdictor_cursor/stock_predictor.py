#!/usr/bin/env python3
"""
Robust Stock Price Prediction System
Predicts stock prices for the next 15 minutes using historical tick data
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

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: TA-Lib not available. Install with: pip install ta")

# Suppress warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """
    A robust stock price prediction system that uses multiple ML models
    to predict future stock prices based on historical tick data.
    """
    
    def __init__(self, db_path: str = "/home/karthik/market.db", model_dir: str = "models"):
        """
        Initialize the StockPredictor
        
        Args:
            db_path: Path to the SQLite database
            model_dir: Directory to save/load trained models
        """
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {}
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        # Always include these models
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        self.models['linear_regression'] = LinearRegression()
        self.models['ridge'] = Ridge(alpha=1.0)
        
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.is_trained = False
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('stock_predictor.log'),
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
    
    def fetch_data(self, symbol: str, limit: int = 10000) -> pd.DataFrame:
        """
        Fetch historical data for a symbol
        
        Args:
            symbol: Stock symbol to fetch data for
            limit: Maximum number of records to fetch
            
        Returns:
            DataFrame with historical data
        """
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
            
            self.logger.info(f"Fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features for prediction
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Basic price features
        df['price_change'] = df['ltp'].diff()
        df['price_change_pct'] = df['ltp'].pct_change()
        df['bid_ask_spread'] = df['ask'] - df['bid']
        df['bid_ask_spread_pct'] = (df['ask'] - df['bid']) / df['ltp']
        df['volume_price_ratio'] = df['vol_traded_today'] / df['ltp']
        
        # Volume features
        df['volume_change'] = df['vol_traded_today'].diff()
        df['volume_change_pct'] = df['vol_traded_today'].pct_change()
        df['buy_sell_ratio'] = df['tot_buy_qty'] / (df['tot_sell_qty'] + 1e-8)
        
        # Price position features
        df['price_vs_high'] = df['ltp'] / df['high']
        df['price_vs_low'] = df['ltp'] / df['low']
        df['price_vs_open'] = df['ltp'] / df['open']
        df['price_vs_prev_close'] = df['ltp'] / df['prev_close']
        
        # Technical indicators using ta library if available
        if TA_AVAILABLE:
            try:
                # Moving averages
                df['sma_5'] = ta.trend.sma_indicator(df['ltp'], window=5)
                df['sma_10'] = ta.trend.sma_indicator(df['ltp'], window=10)
                df['sma_20'] = ta.trend.sma_indicator(df['ltp'], window=20)
                df['ema_5'] = ta.trend.ema_indicator(df['ltp'], window=5)
                df['ema_10'] = ta.trend.ema_indicator(df['ltp'], window=10)
                
                # RSI
                df['rsi'] = ta.momentum.rsi(df['ltp'], window=14)
                
                # Bollinger Bands
                bb = ta.volatility.BollingerBands(df['ltp'], window=20)
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_middle'] = bb.bollinger_mavg()
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_position'] = (df['ltp'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                
                # MACD
                macd = ta.trend.MACD(df['ltp'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_histogram'] = macd.macd_diff()
                
                # Volume indicators
                df['volume_sma'] = df['vol_traded_today'].rolling(window=10).mean()
                df['volume_ratio'] = df['vol_traded_today'] / df['volume_sma']
                
            except Exception as e:
                self.logger.warning(f"Error creating technical indicators: {e}")
        else:
            # Create basic technical indicators manually
            try:
                # Simple moving averages
                df['sma_5'] = df['ltp'].rolling(window=5).mean()
                df['sma_10'] = df['ltp'].rolling(window=10).mean()
                df['sma_20'] = df['ltp'].rolling(window=20).mean()
                
                # Exponential moving averages
                df['ema_5'] = df['ltp'].ewm(span=5).mean()
                df['ema_10'] = df['ltp'].ewm(span=10).mean()
                
                # Simple RSI calculation
                delta = df['ltp'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                df['bb_middle'] = df['ltp'].rolling(window=20).mean()
                bb_std = df['ltp'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
                df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_position'] = (df['ltp'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                
                # Volume indicators
                df['volume_sma'] = df['vol_traded_today'].rolling(window=10).mean()
                df['volume_ratio'] = df['vol_traded_today'] / df['volume_sma']
                
            except Exception as e:
                self.logger.warning(f"Error creating basic technical indicators: {e}")
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'ltp_lag_{lag}'] = df['ltp'].shift(lag)
            df[f'volume_lag_{lag}'] = df['vol_traded_today'].shift(lag)
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'ltp_std_{window}'] = df['ltp'].rolling(window=window).std()
            df[f'ltp_mean_{window}'] = df['ltp'].rolling(window=window).mean()
            df[f'volume_std_{window}'] = df['vol_traded_today'].rolling(window=window).std()
            df[f'volume_mean_{window}'] = df['vol_traded_today'].rolling(window=window).mean()
        
        # Time-based features
        df['hour'] = df['ts'].dt.hour
        df['minute'] = df['ts'].dt.minute
        df['day_of_week'] = df['ts'].dt.dayofweek
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, target_minutes: int = 15) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with target variable (future price)
        
        Args:
            df: DataFrame with features
            target_minutes: Minutes ahead to predict
            
        Returns:
            Tuple of (features, target)
        """
        # Create target variable (price after target_minutes)
        df['target'] = df['ltp'].shift(-target_minutes)
        
        # Remove rows with NaN values in target
        df_clean = df.dropna(subset=['target'])
        
        # Fill remaining NaN values with forward fill, then backward fill, then 0
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        if len(df_clean) < 50:
            raise ValueError(f"Insufficient data for training after cleaning: {len(df_clean)} records")
        
        # Select feature columns (exclude target and metadata)
        exclude_cols = ['symbol', 'ts', 'exch_feed_time', 'last_traded_time', 'target', 'ltp']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        # Remove columns with too many NaN values or constant values
        valid_feature_cols = []
        for col in feature_cols:
            if df_clean[col].isna().sum() < len(df_clean) * 0.5:
                if df_clean[col].nunique() > 1:  # Remove constant columns
                    valid_feature_cols.append(col)
        
        feature_cols = valid_feature_cols
        
        self.feature_columns = feature_cols
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        self.logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, symbol: str = None) -> Dict[str, float]:
        """
        Train multiple models and return their scores
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of model scores
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = {}
        
        for name, model in self.models.items():
            try:
                # Cross-validation score
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_absolute_error')
                scores[name] = -cv_scores.mean()
                
                # Train on full dataset
                model.fit(X_scaled, y)
                
                self.logger.info(f"Trained {name} model with MAE: {scores[name]:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name} model: {e}")
                scores[name] = float('inf')
        
        # Save models and scaler
        if symbol:
            # Save symbol-specific models
            symbol_clean = symbol.replace(':', '_').replace('-', '_')
            symbol_model_dir = self.model_dir / symbol_clean
            symbol_model_dir.mkdir(exist_ok=True)
            
            for name, model in self.models.items():
                joblib.dump(model, symbol_model_dir / f"{name}_model.pkl")
            joblib.dump(self.scaler, symbol_model_dir / "scaler.pkl")
            joblib.dump(self.feature_columns, symbol_model_dir / "feature_columns.pkl")
        else:
            # Save default models
            for name, model in self.models.items():
                joblib.dump(model, self.model_dir / f"{name}_model.pkl")
            joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
            joblib.dump(self.feature_columns, self.model_dir / "feature_columns.pkl")
        
        self.is_trained = True
        return scores
    
    def load_models(self, symbol: str = None) -> bool:
        """Load pre-trained models for a specific symbol"""
        try:
            # Use symbol-specific model directory if symbol provided
            if symbol:
                symbol_clean = symbol.replace(':', '_').replace('-', '_')
                model_dir = self.model_dir / symbol_clean
            else:
                model_dir = self.model_dir
            
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
            
            self.logger.info(f"Successfully loaded pre-trained models for {symbol or 'default'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def predict(self, symbol: str, target_minutes: int = 15) -> Dict:
        """
        Predict future stock price
        
        Args:
            symbol: Stock symbol to predict
            target_minutes: Minutes ahead to predict
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Please train models first.")
        
        try:
            # Fetch recent data
            df = self.fetch_data(symbol, limit=1000)
            
            # Create features
            df_features = self.create_features(df)
            
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
            
            # Calculate ensemble prediction with realistic bounds
            valid_predictions = {k: v for k, v in predictions.items() if v is not None}
            if valid_predictions:
                # Weight models based on their typical performance (exclude linear models for ensemble)
                model_weights = {
                    'xgboost': 0.3,
                    'lightgbm': 0.3,
                    'random_forest': 0.25,
                    'gradient_boosting': 0.15,
                    'linear_regression': 0.0,  # Exclude from ensemble
                    'ridge': 0.0  # Exclude from ensemble
                }
                
                # Calculate weighted average
                weighted_sum = 0
                total_weight = 0
                for model, pred in valid_predictions.items():
                    weight = model_weights.get(model, 0.1)
                    weighted_sum += pred * weight
                    total_weight += weight
                
                if total_weight > 0:
                    ensemble_pred = weighted_sum / total_weight
                else:
                    # Fallback to simple average if no weights
                    ensemble_pred = np.mean(list(valid_predictions.values()))
                
            else:
                raise ValueError("No valid predictions available")
            
            # Get current price and time
            current_price = latest_data['ltp'].iloc[0]
            last_trade_time = latest_data['last_traded_time'].iloc[0]
            
            # Apply realistic bounds to prevent extreme predictions
            ensemble_pred = self._apply_realistic_bounds(ensemble_pred, current_price, target_minutes)
            
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
                'confidence': self._calculate_confidence(predictions, current_price)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {e}")
            raise
    
    def _apply_realistic_bounds(self, predicted_price: float, current_price: float, target_minutes: int) -> float:
        """
        Apply realistic bounds to predictions based on market volatility
        
        Args:
            predicted_price: Raw model prediction
            current_price: Current stock price
            target_minutes: Prediction timeframe in minutes
            
        Returns:
            Bounded prediction price
        """
        # Calculate maximum realistic change based on timeframe
        # For 15 minutes: max 2% change is realistic
        # For 1 hour: max 5% change is realistic
        # For 1 day: max 10% change is realistic
        
        max_change_pct = min(2.0, target_minutes * 0.1)  # 0.1% per minute, max 2%
        max_change = current_price * (max_change_pct / 100)
        
        # Apply bounds
        min_price = current_price - max_change
        max_price = current_price + max_change
        
        # Clamp prediction to realistic bounds
        bounded_price = max(min_price, min(max_price, predicted_price))
        
        # Log if bounds were applied
        if bounded_price != predicted_price:
            self.logger.warning(f"Applied realistic bounds: {predicted_price:.2f} -> {bounded_price:.2f}")
        
        return bounded_price
    
    def _calculate_confidence(self, predictions: Dict, current_price: float) -> float:
        """Calculate prediction confidence based on model agreement"""
        valid_preds = [v for v in predictions.values() if v is not None]
        if len(valid_preds) < 2:
            return 0.5
        
        # Calculate coefficient of variation (lower is better)
        mean_pred = np.mean(valid_preds)
        std_pred = np.std(valid_preds)
        cv = std_pred / mean_pred if mean_pred != 0 else 1
        
        # Convert to confidence score (0-1)
        confidence = max(0, min(1, 1 - cv))
        return confidence
    
    def train_and_predict(self, symbol: str, target_minutes: int = 15) -> Dict:
        """
        Train models and make prediction for a symbol
        
        Args:
            symbol: Stock symbol
            target_minutes: Minutes ahead to predict
            
        Returns:
            Prediction results
        """
        try:
            self.logger.info(f"Starting training and prediction for {symbol}")
            
            # Fetch data
            df = self.fetch_data(symbol, limit=5000)
            
            # Create features
            df_features = self.create_features(df)
            
            # Prepare training data
            X, y = self.prepare_training_data(df_features, target_minutes)
            
            # Train models
            scores = self.train_models(X, y, symbol)
            self.logger.info(f"Model training completed. Scores: {scores}")
            
            # Make prediction
            result = self.predict(symbol, target_minutes)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in train_and_predict for {symbol}: {e}")
            raise


def main():
    """Main function to run the stock predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Price Predictor')
    parser.add_argument('symbol', help='Stock symbol to predict (e.g., NSE:RELIANCE-EQ)')
    parser.add_argument('--minutes', type=int, default=15, help='Minutes ahead to predict (default: 15)')
    parser.add_argument('--retrain', action='store_true', help='Force retrain models')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = StockPredictor()
    
    try:
        if args.retrain or not predictor.load_models():
            # Train models
            result = predictor.train_and_predict(args.symbol, args.minutes)
        else:
            # Use existing models
            result = predictor.predict(args.symbol, args.minutes)
        
        # Display results
        print("\n" + "="*60)
        print("STOCK PRICE PREDICTION RESULTS")
        print("="*60)
        print(f"Symbol: {result['symbol']}")
        print(f"Last Traded Price: ₹{result['current_price']:.2f}")
        print(f"Predicted Price (Next {args.minutes} mins): ₹{result['predicted_price']:.2f}")
        print(f"Expected Change: ₹{result['price_change']:.2f} ({result['price_change_pct']:+.2f}%)")
        print(f"Last Trade Time: {result['last_trade_time']}")
        print(f"Prediction Time: {result['prediction_time']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nIndividual Model Predictions:")
        for model, pred in result['individual_predictions'].items():
            if pred is not None:
                print(f"  {model}: ₹{pred:.2f}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
