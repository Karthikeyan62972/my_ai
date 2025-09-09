#!/bin/bash

echo "🚀 Starting Stock Prediction Monitor Web Viewer..."
echo "📊 Dashboard will be available at: http://localhost:5001"
echo "🔗 API endpoints available at: http://localhost:5001/api/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd /home/karthik/new/perdictor_cursor
python3 web_viewer.py
