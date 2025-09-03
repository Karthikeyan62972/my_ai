import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, ComposedChart } from "recharts";
import dayjs from "dayjs";

export default function AnalyticsPage({ trades, candles }) {
  // Process trades data for analytics
  const processedTrades = trades
    .filter(t => t.pnl !== null && t.pnl !== undefined)
    .map(t => ({
      date: dayjs(t.exit_time || t.entry_time).format('MMM DD'),
      pnl: t.pnl || 0,
      side: t.side,
      symbol: t.symbol,
      qty: t.qty || 0
    }))
    .sort((a, b) => dayjs(a.date, 'MMM DD').diff(dayjs(b.date, 'MMM DD')));

  // Calculate cumulative P&L
  let cumulative = 0;
  const cumulativeData = processedTrades.map(item => {
    cumulative += item.pnl;
    return { ...item, cumulative };
  });

  // Group trades by symbol
  const symbolPerformance = trades.reduce((acc, trade) => {
    if (!acc[trade.symbol]) {
      acc[trade.symbol] = { symbol: trade.symbol, pnl: 0, trades: 0, volume: 0 };
    }
    acc[trade.symbol].pnl += trade.pnl || 0;
    acc[trade.symbol].trades += 1;
    acc[trade.symbol].volume += trade.qty || 0;
    return acc;
  }, {});

  const symbolData = Object.values(symbolPerformance)
    .sort((a, b) => Math.abs(b.pnl) - Math.abs(a.pnl))
    .slice(0, 10);

  // Time-based analysis
  const hourlyPerformance = Array.from({ length: 24 }, (_, hour) => {
    const hourTrades = trades.filter(t => {
      const tradeHour = dayjs(t.entry_time).hour();
      return tradeHour === hour;
    });
    const pnl = hourTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
    return { hour: `${hour}:00`, pnl, trades: hourTrades.length };
  });

  // Risk metrics
  const totalPnL = trades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  const winningTrades = trades.filter(t => (t.pnl || 0) > 0);
  const losingTrades = trades.filter(t => (t.pnl || 0) < 0);
  
  const maxDrawdown = Math.min(...cumulativeData.map(d => d.cumulative));
  const maxProfit = Math.max(...cumulativeData.map(d => d.cumulative));
  const sharpeRatio = calculateSharpeRatio(trades);
  const maxConsecutiveLosses = calculateMaxConsecutiveLosses(trades);

  // Price volatility from candles
  const volatility = calculateVolatility(candles);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">Advanced Analytics</h2>
          <p className="text-slate-400">Deep insights into trading performance and market behavior</p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-white">{trades.length}</div>
          <div className="text-sm text-slate-400">Total Trades Analyzed</div>
        </div>
      </div>

      {/* Risk Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card bg-gradient-to-br from-red-500/20 to-red-600/20 border-red-500/30">
          <div className="text-red-400 text-2xl mb-2">📉</div>
          <div className="text-2xl font-bold text-white">{maxDrawdown.toFixed(2)}</div>
          <div className="text-sm text-red-300">Max Drawdown</div>
        </div>
        
        <div className="card bg-gradient-to-br from-green-500/20 to-green-600/20 border-green-500/30">
          <div className="text-green-400 text-2xl mb-2">📈</div>
          <div className="text-2xl font-bold text-white">{maxProfit.toFixed(2)}</div>
          <div className="text-sm text-green-300">Max Profit</div>
        </div>
        
        <div className="card bg-gradient-to-br from-blue-500/20 to-blue-600/20 border-blue-500/30">
          <div className="text-blue-400 text-2xl mb-2">📊</div>
          <div className="text-2xl font-bold text-white">{sharpeRatio.toFixed(2)}</div>
          <div className="text-sm text-blue-300">Sharpe Ratio</div>
        </div>
        
        <div className="card bg-gradient-to-br from-purple-500/20 to-purple-600/20 border-purple-500/30">
          <div className="text-purple-400 text-2xl mb-2">🎯</div>
          <div className="text-2xl font-bold text-white">{maxConsecutiveLosses}</div>
          <div className="text-sm text-purple-300">Max Consecutive Losses</div>
        </div>
      </div>

      {/* Main Charts Row */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Cumulative P&L Chart */}
        <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30">
          <div className="text-lg font-semibold text-white mb-4">Cumulative P&L Over Time</div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={cumulativeData}>
              <defs>
                <linearGradient id="cumulativeGradient" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.8}/>
                  <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
              <XAxis dataKey="date" stroke="#64748b" tick={{ fill: '#94a3b8' }} />
              <YAxis stroke="#64748b" tick={{ fill: '#94a3b8' }} />
              <Tooltip
                contentStyle={{
                  background: "rgba(15, 23, 42, 0.95)",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                  borderRadius: 12,
                  backdropFilter: "blur(10px)"
                }}
                formatter={(value, name) => [name === 'cumulative' ? `₹${value.toFixed(2)}` : `₹${value.toFixed(2)}`, name === 'cumulative' ? 'Cumulative P&L' : 'Trade P&L']}
              />
              <Area 
                dataKey="cumulative" 
                stroke="#22d3ee" 
                strokeWidth={2}
                fill="url(#cumulativeGradient)" 
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Hourly Performance */}
        <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30">
          <div className="text-lg font-semibold text-white mb-4">Hourly Trading Performance</div>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={hourlyPerformance}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
              <XAxis dataKey="hour" stroke="#64748b" tick={{ fill: '#94a3b8' }} />
              <YAxis yAxisId="left" stroke="#64748b" tick={{ fill: '#94a3b8' }} />
              <YAxis yAxisId="right" orientation="right" stroke="#64748b" tick={{ fill: '#94a3b8' }} />
              <Tooltip
                contentStyle={{
                  background: "rgba(15, 23, 42, 0.95)",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                  borderRadius: 12,
                  backdropFilter: "blur(10px)"
                }}
                formatter={(value, name) => [name === 'pnl' ? `₹${value.toFixed(2)}` : value, name === 'pnl' ? 'P&L' : 'Trades']}
              />
              <Bar yAxisId="left" dataKey="trades" fill="#8b5cf6" opacity={0.7} radius={[2, 2, 0, 0]} />
              <Line yAxisId="right" dataKey="pnl" stroke="#22d3ee" strokeWidth={2} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Symbol Performance */}
      <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30">
        <div className="text-lg font-semibold text-white mb-4">Top 10 Symbols by P&L</div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={symbolData} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
            <XAxis type="number" stroke="#64748b" tick={{ fill: '#94a3b8' }} />
            <YAxis type="category" dataKey="symbol" stroke="#64748b" tick={{ fill: '#94a3b8' }} width={80} />
            <Tooltip
              contentStyle={{
                background: "rgba(15, 23, 42, 0.95)",
                border: "1px solid rgba(148, 163, 184, 0.2)",
                borderRadius: 12,
                backdropFilter: "blur(10px)"
              }}
              formatter={(value, name) => [name === 'pnl' ? `₹${value.toFixed(2)}` : value, name === 'pnl' ? 'P&L' : 'Trades']}
            />
            <Bar dataKey="pnl" fill="#22d3ee" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Win/Loss Distribution */}
        <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30">
          <div className="text-lg font-semibold text-white mb-4">Win/Loss Distribution</div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Winning Trades</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-slate-700 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full" 
                    style={{ width: `${(winningTrades.length / trades.length) * 100}%` }}
                  ></div>
                </div>
                <span className="text-white font-medium">{winningTrades.length}</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Losing Trades</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-slate-700 rounded-full h-2">
                  <div 
                    className="bg-red-500 h-2 rounded-full" 
                    style={{ width: `${(losingTrades.length / trades.length) * 100}%` }}
                  ></div>
                </div>
                <span className="text-white font-medium">{losingTrades.length}</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Win Rate</span>
              <span className="text-white font-medium">
                {trades.length > 0 ? ((winningTrades.length / trades.length) * 100).toFixed(1) : 0}%
              </span>
            </div>
          </div>
        </div>

        {/* Market Volatility */}
        <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30">
          <div className="text-lg font-semibold text-white mb-4">Market Volatility</div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Current Volatility</span>
              <span className="text-white font-medium">{volatility.toFixed(2)}%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Price Range</span>
              <span className="text-white font-medium">
                {candles.length > 0 ? 
                  `₹${Math.min(...candles.map(c => c.l)).toFixed(2)} - ₹${Math.max(...candles.map(c => c.h)).toFixed(2)}` 
                  : 'N/A'
                }
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Data Points</span>
              <span className="text-white font-medium">{candles.length}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper functions
function calculateSharpeRatio(trades) {
  if (trades.length === 0) return 0;
  
  const returns = trades.map(t => t.pnl || 0);
  const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
  const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
  const stdDev = Math.sqrt(variance);
  
  return stdDev > 0 ? avgReturn / stdDev : 0;
}

function calculateMaxConsecutiveLosses(trades) {
  let maxConsecutive = 0;
  let currentConsecutive = 0;
  
  trades.forEach(trade => {
    if ((trade.pnl || 0) < 0) {
      currentConsecutive++;
      maxConsecutive = Math.max(maxConsecutive, currentConsecutive);
    } else {
      currentConsecutive = 0;
    }
  });
  
  return maxConsecutive;
}

function calculateVolatility(candles) {
  if (candles.length < 2) return 0;
  
  const returns = [];
  for (let i = 1; i < candles.length; i++) {
    const prevClose = candles[i-1].c;
    const currentClose = candles[i].c;
    if (prevClose > 0) {
      returns.push((currentClose - prevClose) / prevClose);
    }
  }
  
  if (returns.length === 0) return 0;
  
  const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
  const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
  const stdDev = Math.sqrt(variance);
  
  return stdDev * 100; // Convert to percentage
}
