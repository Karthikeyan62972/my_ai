# Stock Price Predictor

A robust machine learning system for predicting stock prices using historical tick data from SQLite database.

## Features

- **Multiple ML Models**: Uses ensemble of XGBoost, Random Forest, Gradient Boosting, and Linear models
- **Technical Indicators**: Comprehensive feature engineering with moving averages, RSI, Bollinger Bands, MACD
- **Robust Error Handling**: Comprehensive logging and error recovery
- **Database Integration**: Direct SQLite database connectivity
- **Real-time Predictions**: Predicts prices for next 15 minutes (configurable)
- **Confidence Scoring**: Provides prediction confidence based on model agreement

## Requirements

### Core Dependencies
```bash
pip install numpy pandas scikit-learn joblib
```

### Optional Dependencies (for enhanced performance)
```bash
pip install xgboost lightgbm ta matplotlib seaborn
```

## Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python3 run_prediction.py NSE:RELIANCE-EQ 15
```

### Advanced Usage
```bash
python3 stock_predictor.py NSE:RELIANCE-EQ --minutes 15 --retrain
```

### Available Symbols
The system works with any symbol in your database. Top symbols by volume:
- NSE:NIFTY50-INDEX
- NSE:TATASTEEL-EQ
- NSE:INFY-EQ
- NSE:HDFCBANK-EQ
- NSE:ICICIBANK-EQ
- NSE:RELIANCE-EQ
- NSE:TATAMOTORS-EQ
- NSE:BAJFINANCE-EQ
- NSE:M&M-EQ
- NSE:TCS-EQ

## Database Schema

The system expects a SQLite database with a `ticks` table containing:
- `symbol`: Stock symbol
- `ts`: Timestamp
- `ltp`: Last traded price
- `bid`, `ask`: Bid/ask prices
- `vol_traded_today`: Volume traded
- And other market data fields

## Output Format

```
======================================================================
📈 STOCK PRICE PREDICTION RESULTS
======================================================================
🏷️  Symbol: NSE:RELIANCE-EQ
💰 Current Price: ₹1379.40
🔮 Predicted Price (Next 15 mins): ₹1379.45
📈 Expected Change: +₹0.05 (+0.00%)
⏰ Last Trade Time: 2025-09-08 10:10:41
🕐 Prediction Time: 2025-09-09 00:17:42
🎯 Confidence: 83.9%

🤖 Individual Model Predictions:
   Xgboost: ₹1379.45
   Random Forest: ₹1379.39
   Gradient Boosting: ₹1379.52
   Linear Regression: ₹1269.64
   Ridge: ₹1931.87
======================================================================
```

## Model Performance

The system uses ensemble learning with the following model weights:
- XGBoost: 30%
- LightGBM: 30% (if available)
- Random Forest: 25%
- Gradient Boosting: 15%
- Linear models: Excluded from ensemble (used for comparison)

## Configuration

### Database Path
Default: `/home/karthik/market.db`
Modify in `StockPredictor` class initialization.

### Model Directory
Default: `models/`
Trained models are saved here for reuse.

### Prediction Timeframe
Default: 15 minutes
Can be configured via command line argument.

## Logging

The system creates detailed logs in `stock_predictor.log` including:
- Training progress
- Model performance metrics
- Error messages
- Prediction results

## Error Handling

The system includes comprehensive error handling for:
- Database connection issues
- Insufficient data
- Model training failures
- Prediction errors
- Missing dependencies

## Performance Tips

1. **Install XGBoost and LightGBM** for better accuracy
2. **Use sufficient data** (minimum 1000 records recommended)
3. **Regular retraining** for better predictions
4. **Monitor confidence scores** - low confidence indicates unreliable predictions

## Troubleshooting

### Common Issues

1. **"No module named 'xgboost'"**
   ```bash
   pip install xgboost
   ```

2. **"Insufficient data for training"**
   - Ensure your database has enough historical data
   - Check symbol name is correct

3. **"Database connection error"**
   - Verify database path is correct
   - Check database permissions

4. **Low prediction confidence**
   - Retrain models with more recent data
   - Check for data quality issues

## License

This project is for educational and research purposes. Use at your own risk for actual trading decisions.

## ⚠️ Safety Features

The system includes multiple safety features to prevent unrealistic predictions:

- **Realistic Bounds**: Maximum 2% price change for 15-minute predictions
- **Symbol-Specific Models**: Each stock has its own trained model for accuracy
- **Automatic Correction**: Extreme predictions are automatically bounded
- **Confidence Scoring**: Low confidence warnings when predictions are unreliable

## Disclaimer

This software is for educational purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for trading decisions. Always consult with financial professionals before making investment decisions.

**The system now provides realistic predictions only and is safe for educational use.**
