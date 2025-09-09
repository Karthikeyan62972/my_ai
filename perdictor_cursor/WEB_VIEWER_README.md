# 🌐 Web Database Viewer

A beautiful, modern web-based interface for monitoring your stock prediction system in real-time.

## ✨ Features

### 📊 **Real-time Dashboard**
- **Live Statistics**: Total predictions, confidence levels, signal distribution
- **Auto-refresh**: Updates every 30 seconds automatically
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### 📈 **Interactive Data Views**
- **Recent Predictions**: Filter by time range, symbol, and limit
- **Active Triggers**: Monitor ongoing trading signals
- **Performance Charts**: Visual representation of prediction trends
- **Symbol-specific Stats**: Detailed analysis for individual stocks

### 🎨 **Modern UI/UX**
- **Beautiful Design**: Gradient backgrounds, smooth animations, modern cards
- **Color-coded Signals**: BUY (green), SELL (red), HOLD (yellow), AVOID (gray)
- **Confidence Bars**: Visual confidence indicators
- **Status Badges**: Clear trigger status indicators

### 🔧 **Advanced Controls**
- **Time Range Filtering**: 1 hour, 6 hours, 24 hours
- **Symbol Filtering**: View specific stocks or all symbols
- **Limit Controls**: Adjust number of records displayed
- **Real-time Search**: Instant filtering and sorting

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
cd /home/karthik/new/perdictor_cursor
pip3 install -r web_requirements.txt
```

### 2. **Start the Web Server**
```bash
./start_web_viewer.sh
```

### 3. **Access the Dashboard**
Open your browser and go to: **http://localhost:5001**

## 📱 Usage Guide

### **Main Dashboard**
- **Statistics Cards**: Overview of system performance
- **Recent Predictions Table**: Latest predictions with filtering options
- **Active Triggers Table**: Current trading signals being monitored
- **Performance Chart**: Visual trend analysis

### **Filtering Options**
- **Time Range**: Select how far back to look (1h, 6h, 24h)
- **Symbol**: Choose specific stock or view all
- **Limit**: Control number of records (25, 50, 100)

### **Data Interpretation**
- **Confidence**: Higher percentage = more reliable prediction
- **Signals**: BUY/SELL/HOLD/AVOID based on AI analysis
- **Price Change %**: Expected price movement
- **Status**: MONITORING/SUCCESS/FAILURE for triggers

## 🔗 API Endpoints

The web viewer also provides REST API endpoints:

### **Statistics**
```
GET /api/stats?hours=24
```
Returns overall system statistics for specified time range.

### **Predictions**
```
GET /api/predictions?hours=1&limit=50&symbol=NSE:RELIANCE-EQ
```
Returns recent predictions with optional filtering.

### **Triggers**
```
GET /api/triggers
```
Returns all active trading triggers.

### **Symbols**
```
GET /api/symbols
```
Returns list of all monitored symbols.

### **Symbol Stats**
```
GET /api/symbol/NSE:RELIANCE-EQ?hours=24
```
Returns detailed statistics for a specific symbol.

## 🎯 Key Benefits

### **For Traders**
- **Visual Monitoring**: Easy-to-read charts and tables
- **Real-time Updates**: Stay informed of latest predictions
- **Risk Management**: Clear confidence and risk indicators
- **Mobile Access**: Monitor from anywhere

### **For Analysis**
- **Historical Data**: Track prediction accuracy over time
- **Performance Metrics**: Success rates and confidence trends
- **Symbol Comparison**: Compare different stocks side-by-side
- **Export Ready**: Data available via API for further analysis

## 🔧 Technical Details

### **Architecture**
- **Backend**: Flask (Python) with SQLite database
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Charts**: Chart.js for interactive visualizations
- **Styling**: Modern CSS with gradients and animations

### **Database Integration**
- **Monitoring DB**: `/home/karthik/monitoring.db`
- **Market DB**: `/home/karthik/market.db`
- **Real-time Queries**: Optimized SQL with proper indexing
- **Error Handling**: Graceful fallbacks for database issues

### **Performance**
- **Auto-refresh**: 30-second intervals
- **Efficient Queries**: Limited result sets with pagination
- **Caching**: Browser-side caching for better performance
- **Responsive**: Optimized for all screen sizes

## 🛠️ Customization

### **Styling**
Edit `templates/dashboard.html` to customize:
- Colors and themes
- Layout and spacing
- Fonts and typography
- Animation effects

### **Functionality**
Modify `web_viewer.py` to add:
- New API endpoints
- Additional data processing
- Custom filters
- Export functionality

### **Charts**
Update Chart.js configuration for:
- Different chart types
- Custom data visualization
- Interactive features
- Export options

## 🔍 Troubleshooting

### **Common Issues**

**Web server won't start:**
```bash
# Check if port 5000 is in use
netstat -tulpn | grep :5000

# Kill existing processes
pkill -f web_viewer.py

# Start fresh
./start_web_viewer.sh
```

**Database connection errors:**
```bash
# Check database permissions
ls -la /home/karthik/monitoring.db

# Verify database integrity
sqlite3 /home/karthik/monitoring.db "PRAGMA integrity_check;"
```

**No data showing:**
```bash
# Check if monitoring system is running
ps aux | grep multi_symbol_monitor

# Verify database has data
sqlite3 /home/karthik/monitoring.db "SELECT COUNT(*) FROM prediction_logs;"
```

## 📊 Sample Data

The dashboard shows:
- **189 Total Predictions** (last 24 hours)
- **38 Symbols Monitored**
- **98.6% Average Confidence**
- **11 BUY, 14 SELL, 164 HOLD signals**
- **0 Active Triggers** (no strong signals yet)

## 🎉 Success!

Your web-based monitoring dashboard is now ready! 

**Access it at: http://localhost:5001**

The interface provides a beautiful, real-time view of your AI-powered stock prediction system with all the data you need to make informed trading decisions.

---

*Happy Trading! 📈💰*
