# 🚀 MULTI-SYMBOL MONITORING SYSTEM

## 🎯 **SYSTEM OVERVIEW**

Your new multi-symbol monitoring system automatically tracks **38 symbols** from your `symbols.txt` file and stores all predictions in a separate database for analysis and profit tracking.

### 📊 **What It Does:**

1. **Monitors All Symbols**: Automatically processes all 38 symbols from your symbols.txt
2. **Stores Predictions**: Every prediction is saved to `/home/karthik/monitoring.db`
3. **Creates Trading Triggers**: Strong signals (confidence ≥75% + profit ≥1%) are tracked
4. **Tracks Outcomes**: Monitors if predictions come true after 15 minutes
5. **Real-Time Updates**: Continuously syncs with your Fyers data

## 🗄️ **DATABASE STRUCTURE**

### **Main Database**: `/home/karthik/market.db`
- Contains live tick data from your Fyers script
- Used for making predictions

### **Monitoring Database**: `/home/karthik/monitoring.db`
- **`prediction_logs`**: All predictions with timestamps
- **`trading_triggers`**: Strong signals being monitored

## 📋 **USAGE COMMANDS**

### **1. Run Once (Test Mode)**
```bash
python3 multi_symbol_monitor.py --once --minutes 15
```
- Processes all 38 symbols once
- Trains models if needed
- Stores predictions in database

### **2. Continuous Monitoring (Live Trading)**
```bash
python3 multi_symbol_monitor.py --minutes 15 --interval 30
```
- Monitors all symbols every 30 seconds
- Perfect for live market hours
- Automatically creates trading triggers

### **3. View Statistics**
```bash
python3 database_viewer.py --stats --hours 24
```
- Shows overall performance statistics
- Success rates, signal counts, etc.

### **4. View Recent Predictions**
```bash
python3 database_viewer.py --recent --hours 1 --limit 20
```
- Shows last 20 predictions from past hour

### **5. View Active Triggers**
```bash
python3 database_viewer.py --active
```
- Shows currently monitoring trading triggers

### **6. View Completed Triggers**
```bash
python3 database_viewer.py --completed --days 1
```
- Shows completed triggers with success/failure outcomes

### **7. Symbol-Specific Stats**
```bash
python3 database_viewer.py --symbol NSE:RELIANCE-EQ --hours 24
```
- Shows detailed stats for a specific symbol

## 🎯 **TRADING TRIGGER SYSTEM**

### **Strong Signal Criteria:**
- **Confidence**: ≥ 75%
- **Profit Potential**: ≥ 1.0%
- **Signal**: BUY or SELL (not HOLD)

### **Example Trigger:**
```
Symbol: NSE:RELIANCE-EQ
Current Price: ₹1500.00
Predicted Price: ₹1520.00 (15 mins)
Signal: BUY
Confidence: 85%
Status: monitoring
```

### **Outcome Tracking:**
- **Success**: Actual price reaches predicted price within 15 minutes
- **Failure**: Price doesn't reach target
- **Status**: Updated automatically after 15 minutes

## 📊 **MONITORING WORKFLOW**

### **During Market Hours:**
1. **Your Fyers Script**: Updates `/home/karthik/market.db` with live data
2. **Multi-Symbol Monitor**: Reads new data every 30 seconds
3. **Predictions**: Made for all 38 symbols
4. **Storage**: All predictions saved to monitoring database
5. **Triggers**: Strong signals create monitoring entries
6. **Outcomes**: Tracked automatically after 15 minutes

### **Data Flow:**
```
Fyers API → market.db → Multi-Symbol Monitor → monitoring.db → Trading Triggers
```

## 🚀 **LIVE TRADING SETUP**

### **Step 1: Start Continuous Monitoring**
```bash
# Run this during market hours (9:15 AM - 3:30 PM IST)
python3 multi_symbol_monitor.py --minutes 15 --interval 30
```

### **Step 2: Monitor Active Triggers**
```bash
# Check for new trading opportunities
python3 database_viewer.py --active
```

### **Step 3: Check Performance**
```bash
# Review success rates
python3 database_viewer.py --stats --hours 4
```

## 📈 **PROFIT TRACKING**

### **View Your Success Rate:**
```bash
python3 database_viewer.py --stats --hours 24
```

### **Example Output:**
```
📊 OVERALL MONITORING STATISTICS (Last 24 hours)
============================================================
📈 PREDICTIONS:
   Total: 1,248
   Symbols: 38
   Avg Confidence: 78.5%
   BUY: 156
   SELL: 89
   HOLD: 1,003

🎯 TRADING TRIGGERS:
   Total: 45
   Active: 3
   Successful: 28
   Failed: 14
   Success Rate: 66.7%
```

## 🎯 **SYMBOLS BEING MONITORED**

Your system monitors these 38 symbols:
- NSE:NIFTY50-INDEX
- NSE:SBIN-EQ, NSE:RELIANCE-EQ, NSE:TCS-EQ
- NSE:HDFCBANK-EQ, NSE:ICICIBANK-EQ, NSE:INFY-EQ
- NSE:AXISBANK-EQ, NSE:KOTAKBANK-EQ, NSE:ITC-EQ
- NSE:HINDUNILVR-EQ, NSE:LT-EQ, NSE:MARUTI-EQ
- NSE:M&M-EQ, NSE:BAJFINANCE-EQ, NSE:BAJAJFINSV-EQ
- NSE:ADANIENT-EQ, NSE:ADANIPORTS-EQ, NSE:POWERGRID-EQ
- NSE:ONGC-EQ, NSE:NTPC-EQ, NSE:TATAMOTORS-EQ
- NSE:TATASTEEL-EQ, NSE:TITAN-EQ, NSE:WIPRO-EQ
- NSE:TECHM-EQ, NSE:ULTRACEMCO-EQ, NSE:GRASIM-EQ
- NSE:JSWSTEEL-EQ, NSE:COALINDIA-EQ, NSE:BPCL-EQ
- NSE:IOC-EQ, NSE:HEROMOTOCO-EQ, NSE:BRITANNIA-EQ
- NSE:HCLTECH-EQ, NSE:CIPLA-EQ, NSE:DIVISLAB-EQ
- NSE:DRREDDY-EQ

## ⚡ **PERFORMANCE BENCHMARKS**

### **Training Performance:**
- **NSE:NIFTY50-INDEX**: XGBoost MAE: 0.6475
- **NSE:SBIN-EQ**: XGBoost MAE: 0.0435
- **NSE:RELIANCE-EQ**: XGBoost MAE: 0.0600
- **NSE:TCS-EQ**: XGBoost MAE: 0.1042
- **NSE:HDFCBANK-EQ**: XGBoost MAE: 0.0374

### **Speed Performance:**
- **Database Fetch**: 0.023-0.133 seconds per symbol
- **Model Training**: ~1-2 seconds per symbol
- **Total Cycle**: ~2-3 minutes for all 38 symbols

## 🔧 **CUSTOMIZATION**

### **Adjust Strong Signal Criteria:**
Edit `multi_symbol_monitor.py`:
```python
self.strong_signal_criteria = {
    'min_confidence': 0.75,  # 75% confidence
    'min_profit_pct': 1.0,   # 1% profit potential
    'min_volume_ratio': 1.2  # 20% above average volume
}
```

### **Change Monitoring Interval:**
```bash
# Monitor every 15 seconds (more frequent)
python3 multi_symbol_monitor.py --minutes 15 --interval 15

# Monitor every 60 seconds (less frequent)
python3 multi_symbol_monitor.py --minutes 15 --interval 60
```

## 📊 **DATABASE QUERIES**

### **Direct Database Access:**
```bash
# View prediction logs
sqlite3 /home/karthik/monitoring.db "SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT 10;"

# View trading triggers
sqlite3 /home/karthik/monitoring.db "SELECT * FROM trading_triggers WHERE status='monitoring';"

# View success rate
sqlite3 /home/karthik/monitoring.db "SELECT COUNT(*) as total, SUM(CASE WHEN outcome='success' THEN 1 ELSE 0 END) as successful FROM trading_triggers WHERE status='completed';"
```

## 🚨 **IMPORTANT NOTES**

1. **Database Separation**: Monitoring data is stored separately from market data
2. **No Locking Issues**: Your Fyers script won't interfere with monitoring
3. **Automatic Cleanup**: Old predictions are automatically managed
4. **Real-Time Sync**: Always uses latest data from your Fyers script
5. **Error Handling**: Robust error handling for individual symbol failures

## 🎯 **NEXT STEPS**

1. **Test During Market Hours**: Run continuous monitoring during live market
2. **Analyze Success Rates**: Track which symbols perform best
3. **Optimize Criteria**: Adjust strong signal thresholds based on results
4. **Scale Up**: Add more symbols to symbols.txt as needed
5. **Automate Trading**: Use triggers to automatically place orders

**Your multi-symbol monitoring system is now ready for live intraday trading! 🚀💰**

