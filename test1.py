#!/usr/bin/env python3
"""
Robust Stock Price Prediction Script - SQLite Only
Predicts NSE:DRREDDY-EQ stock price for next 15 minutes using ML models
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import warnings
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import os
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPricePredictor:
    def __init__(self, db_path):
        """
        Initialize the Stock Price Predictor
        
        Args:
            db_path (str): Path to SQLite database
        """
        self.db_path = db_path
        self.symbol = 'NSE:DRREDDY-EQ'
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.prediction_horizon = 15  # minutes
        
    def connect_database(self):
        """Establish SQLite database connection"""
        try:
            return sqlite3.connect(self.db_path)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def fetch_data(self):
        """Fetch tick data from database"""
        
        # First, let's check the actual latest record in the database
        latest_query = """
        SELECT ts, ltp, last_traded_time 
        FROM ticks 
        WHERE symbol = ? 
        ORDER BY ts DESC 
        LIMIT 1
        """
        
        conn = self.connect_database()
        latest_record = pd.read_sql_query(latest_query, conn, params=(self.symbol,))
        if not latest_record.empty:
            logger.info(f"Database latest record - ts: {latest_record.iloc[0]['ts']}, ltp: {latest_record.iloc[0]['ltp']}, last_traded_time: {latest_record.iloc[0]['last_traded_time']}")
        
        # Main data query
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
        """
        
        try:
            df = pd.read_sql_query(query, conn, params=(self.symbol,))
            conn.close()
            
            if df.empty:
                raise ValueError(f"No data found for symbol: {self.symbol}")
            
            logger.info(f"Fetched {len(df)} records for {self.symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Data fetching failed: {e}")
            raise

    def preprocess_data(self, df):
        """Preprocess the fetched data"""
        logger.info("Preprocessing data...")
        
        # Convert timestamp columns to datetime
        timestamp_columns = ['ts', 'exch_feed_time', 'last_traded_time']
        for col in timestamp_columns:
            if col in df.columns:
                # Handle different timestamp formats
                if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                    # Assume epoch timestamp - try different units
                    try:
                        # Try nanoseconds first
                        df[col] = pd.to_datetime(df[col], unit='ns', errors='coerce')
                        # If all dates are before 1990, try milliseconds
                        if df[col].max() < pd.Timestamp('1990-01-01'):
                            df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce')
                        # If still before 1990, try seconds
                        if df[col].max() < pd.Timestamp('1990-01-01'):
                            df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')
                    except:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Sort by timestamp for chronological processing
        df = df.sort_values('ts').reset_index(drop=True)
        
        # Handle missing values using pandas forward fill
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill')
        
        # Fill any remaining NaN values with median
        imputer = SimpleImputer(strategy='median')
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        # Create additional features
        df = self.create_features(df)
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        logger.info(f"Data preprocessed. Shape: {df.shape}")
        return df

    def create_features(self, df):
        """Create technical indicators and additional features"""
        logger.info("Creating technical features...")
        
        # Price-based features
        df['spread'] = df['ask'] - df['bid']
        df['mid_price'] = (df['ask'] + df['bid']) / 2
        df['price_change'] = df['ltp'].pct_change()
        df['volume_change'] = df['vol_traded_today'].pct_change()
        
        # Moving averages
        windows = [5, 10, 20, 50]
        for window in windows:
            df[f'ma_{window}'] = df['ltp'].rolling(window=window, min_periods=1).mean()
            df[f'vol_ma_{window}'] = df['vol_traded_today'].rolling(window=window, min_periods=1).mean()
        
        # Volatility features
        df['volatility_5'] = df['ltp'].rolling(window=5, min_periods=1).std()
        df['volatility_20'] = df['ltp'].rolling(window=20, min_periods=1).std()
        
        # Order book features
        df['order_imbalance'] = (df['tot_buy_qty'] - df['tot_sell_qty']) / (df['tot_buy_qty'] + df['tot_sell_qty'] + 1e-8)
        df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
        
        # Time-based features
        df['hour'] = df['ts'].dt.hour
        df['minute'] = df['ts'].dt.minute
        df['day_of_week'] = df['ts'].dt.dayofweek
        
        # Lag features
        lag_periods = [1, 2, 3, 5, 10]
        for lag in lag_periods:
            df[f'ltp_lag_{lag}'] = df['ltp'].shift(lag)
            df[f'volume_lag_{lag}'] = df['vol_traded_today'].shift(lag)
        
        return df

    def prepare_training_data(self, df):
        """Prepare data for training"""
        logger.info("Preparing training data...")
        
        # Define feature columns (exclude non-predictive columns)
        exclude_columns = ['symbol', 'ts', 'exch_feed_time', 'last_traded_time']
        self.feature_columns = [col for col in df.columns if col not in exclude_columns and col != 'ltp']
        
        # Prepare features and target
        X = df[self.feature_columns].copy()
        y = df['ltp'].copy()
        
        # Create future price target (next 15 minutes approximation)
        # Since we don't have exact 15-minute intervals, we'll use next few ticks as proxy
        y_future = df['ltp'].shift(-10)  # Approximate future price
        
        # Remove rows with NaN targets
        mask = ~y_future.isna()
        X = X[mask]
        y = y[mask]
        y_future = y_future[mask]
        
        # Use future price as target
        y = y_future
        
        # Handle any remaining NaN values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        logger.info(f"Training data shape - X: {X_imputed.shape}, y: {y.shape}")
        return X_imputed, y, df

    def train_models(self, X, y):
        """Train multiple ML models"""
        logger.info("Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['feature_scaler'] = scaler
        
        # Define models
        models_config = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        # Train models and evaluate
        best_model = None
        best_score = float('inf')
        
        for name, model in models_config.items():
            logger.info(f"Training {name}...")
            
            if name == 'linear_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"{name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # Store model
            self.models[name] = model
            
            # Track best model
            if mae < best_score:
                best_score = mae
                best_model = name
        
        logger.info(f"Best model: {best_model} with MAE: {best_score:.4f}")
        return best_model

    def predict_future_price(self, df, model_name='random_forest'):
        """Predict future stock price"""
        logger.info(f"Making prediction using {model_name}...")
        
        # Get the latest data point
        latest_data = df.iloc[-1:][self.feature_columns].copy()
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        latest_data_imputed = pd.DataFrame(
            imputer.fit_transform(latest_data), 
            columns=latest_data.columns, 
            index=latest_data.index
        )
        
        # Scale if using linear regression
        if model_name == 'linear_regression':
            latest_data_scaled = self.scalers['feature_scaler'].transform(latest_data_imputed)
            prediction = self.models[model_name].predict(latest_data_scaled)[0]
        else:
            prediction = self.models[model_name].predict(latest_data_imputed)[0]
        
        return prediction

    def run_prediction(self):
        """Main function to run the complete prediction pipeline"""
        try:
            logger.info("="*50)
            logger.info("STOCK PRICE PREDICTION SYSTEM")
            logger.info("="*50)
            
            # Fetch and preprocess data
            raw_data = self.fetch_data()
            processed_data = self.preprocess_data(raw_data)
            
            # Prepare training data
            X, y, full_data = self.prepare_training_data(processed_data)
            
            if len(X) < 50:
                logger.warning("Insufficient data for reliable prediction. Need at least 50 records.")
                return None
            
            # Train models
            best_model = self.train_models(X, y)
            
            # Get latest information
            latest_record = full_data.iloc[-1]
            last_traded_price = latest_record['ltp']
            last_traded_time = latest_record['last_traded_time']
            
            # Debug: Print the last few records to verify data
            logger.info("Debug: Last 3 records from database:")
            debug_cols = ['ts', 'ltp', 'last_traded_time']
            for i in range(max(0, len(full_data)-3), len(full_data)):
                record = full_data.iloc[i]
                logger.info(f"  Record {i}: ts={record['ts']}, ltp={record['ltp']}, last_traded_time={record['last_traded_time']}")
            
            logger.info(f"Selected latest record - LTP: {last_traded_price}, Time: {last_traded_time}")
            
            # Make prediction
            predicted_price = self.predict_future_price(full_data, best_model)
            
            # Calculate prediction time (15 minutes from last trade)
            if pd.notna(last_traded_time) and last_traded_time > pd.Timestamp('1990-01-01'):
                prediction_time = last_traded_time + timedelta(minutes=15)
            else:
                # Use current time if timestamp is invalid
                current_time = datetime.now()
                prediction_time = current_time + timedelta(minutes=15)
                last_traded_time = current_time
                logger.warning("Invalid last_traded_time, using current time")
            
            # Print results
            print("\n" + "="*60)
            print("STOCK PRICE PREDICTION RESULTS")
            print("="*60)
            print(f"Symbol: {self.symbol}")
            print(f"Last Traded Price: ₹{last_traded_price:.2f}")
            print(f"Last Traded Time: {last_traded_time}")
            print("-"*60)
            print(f"Predicted Price (15 min): ₹{predicted_price:.2f}")
            print(f"Prediction Time: {prediction_time}")
            print(f"Expected Change: ₹{predicted_price - last_traded_price:.2f}")
            print(f"Expected Change %: {((predicted_price - last_traded_price) / last_traded_price) * 100:.2f}%")
            print(f"Best Model Used: {best_model}")
            print("="*60)
            
            # Additional insights
            change_direction = "UP" if predicted_price > last_traded_price else "DOWN"
            confidence = "HIGH" if abs(predicted_price - last_traded_price) > (last_traded_price * 0.001) else "LOW"
            
            print(f"Direction: {change_direction}")
            print(f"Confidence: {confidence}")
            print("="*60)
            
            return {
                'symbol': self.symbol,
                'last_traded_price': actual_latest_ltp,
                'last_traded_time': last_traded_time,
                'predicted_price': predicted_price,
                'prediction_time': prediction_time,
                'expected_change': predicted_price - actual_latest_ltp,
                'expected_change_pct': ((predicted_price - actual_latest_ltp) / actual_latest_ltp) * 100,
                'best_model': best_model,
                'direction': change_direction,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

def main():
    """Main execution function"""
    
    # Your SQLite database path
    db_path = '/home/karthik/market.db'
    
    try:
        # Initialize predictor
        predictor = StockPricePredictor(db_path)
        
        # Run prediction
        result = predictor.run_prediction()
        
        if result:
            logger.info("Prediction completed successfully!")
        else:
            logger.error("Prediction failed!")
            
    except KeyboardInterrupt:
        logger.info("Prediction interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()