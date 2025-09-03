import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid } from "recharts";

export default function OverviewPage({ trades, predictions, metrics }) {
  // Calculate trading statistics
  const totalTrades = trades.length;
  const winningTrades = trades.filter(t => (t.pnl || 0) > 0).length;
  const losingTrades = trades.filter(t => (t.pnl || 0) < 0).length;
  const breakEvenTrades = totalTrades - winningTrades - losingTrades;
  
  const totalPnL = trades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  const avgPnL = totalTrades > 0 ? totalPnL / totalTrades : 0;
  
  // Prediction accuracy data
  const totalPredictions = predictions.length;
  const correctPredictions = predictions.filter(p => p.hit === 1).length;
  const wrongPredictions = predictions.filter(p => p.hit === 0).length;
  const pendingPredictions = totalPredictions - correctPredictions - wrongPredictions;

  // Trading performance by side
  const buyTrades = trades.filter(t => t.side === 'BUY');
  const sellTrades = trades.filter(t => t.side === 'SELL');
  const buyPnL = buyTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  const sellPnL = sellTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);

  // Pie chart data
  const tradeResultsData = [
    { name: 'Winning', value: winningTrades, color: '#10b981' },
    { name: 'Losing', value: losingTrades, color: '#ef4444' },
    { name: 'Break Even', value: breakEvenTrades, color: '#6b7280' }
  ].filter(item => item.value > 0);

  const predictionAccuracyData = [
    { name: 'Correct', value: correctPredictions, color: '#10b981' },
    { name: 'Wrong', value: wrongPredictions, color: '#ef4444' },
    { name: 'Pending', value: pendingPredictions, color: '#f59e0b' }
  ].filter(item => item.value > 0);

  const sidePerformanceData = [
    { name: 'BUY', value: buyTrades.length, color: '#3b82f6' },
    { name: 'SELL', value: sellTrades.length, color: '#8b5cf6' }
  ].filter(item => item.value > 0);

  // Monthly performance data
  const monthlyData = [
    { month: 'Jan', pnl: 1200, trades: 15 },
    { month: 'Feb', pnl: -800, trades: 12 },
    { month: 'Mar', pnl: 2100, trades: 18 },
    { month: 'Apr', pnl: 1800, trades: 16 },
    { month: 'May', pnl: 950, trades: 14 },
    { month: 'Jun', pnl: 1600, trades: 17 }
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">Market Overview</h2>
          <p className="text-slate-400">Comprehensive trading insights and performance metrics</p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-white">₹{totalPnL.toFixed(2)}</div>
          <div className="text-sm text-slate-400">Total P&L</div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card bg-gradient-to-br from-blue-500/20 to-blue-600/20 border-blue-500/30">
          <div className="text-blue-400 text-2xl mb-2">📊</div>
          <div className="text-2xl font-bold text-white">{totalTrades}</div>
          <div className="text-sm text-blue-300">Total Trades</div>
        </div>
        
        <div className="card bg-gradient-to-br from-green-500/20 to-green-600/20 border-green-500/30">
          <div className="text-green-400 text-2xl mb-2">🎯</div>
          <div className="text-2xl font-bold text-white">{totalPredictions}</div>
          <div className="text-sm text-green-300">Predictions</div>
        </div>
        
        <div className="card bg-gradient-to-br from-purple-500/20 to-purple-600/20 border-purple-500/30">
          <div className="text-purple-400 text-2xl mb-2">📈</div>
          <div className="text-2xl font-bold text-white">{totalPredictions > 0 ? ((correctPredictions / totalPredictions) * 100).toFixed(1) : 0}%</div>
          <div className="text-sm text-purple-300">Accuracy Rate</div>
        </div>
        
        <div className="card bg-gradient-to-br from-yellow-500/20 to-yellow-600/20 border-yellow-500/30">
          <div className="text-yellow-400 text-2xl mb-2">💰</div>
          <div className="text-2xl font-bold text-white">₹{avgPnL.toFixed(2)}</div>
          <div className="text-sm text-yellow-300">Avg P&L per Trade</div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trade Results Pie Chart */}
        <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30">
          <div className="text-lg font-semibold text-white mb-4">Trade Results Distribution</div>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={tradeResultsData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="value"
              >
                {tradeResultsData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: "rgba(15, 23, 42, 0.95)",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                  borderRadius: 12,
                  backdropFilter: "blur(10px)"
                }}
                formatter={(value, name) => [value, name]}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Prediction Accuracy Pie Chart */}
        <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30">
          <div className="text-lg font-semibold text-white mb-4">Prediction Accuracy</div>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={predictionAccuracyData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="value"
              >
                {predictionAccuracyData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: "rgba(15, 23, 42, 0.95)",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                  borderRadius: 12,
                  backdropFilter: "blur(10px)"
                }}
                formatter={(value, name) => [value, name]}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Additional Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trading Side Performance */}
        <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30">
          <div className="text-lg font-semibold text-white mb-4">Trading Side Performance</div>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={sidePerformanceData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="value"
              >
                {sidePerformanceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: "rgba(15, 23, 42, 0.95)",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                  borderRadius: 12,
                  backdropFilter: "blur(10px)"
                }}
                formatter={(value, name) => [value, name]}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-4 grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-lg font-bold text-blue-400">₹{buyPnL.toFixed(2)}</div>
              <div className="text-sm text-slate-400">BUY P&L</div>
            </div>
            <div>
              <div className="text-lg font-bold text-purple-400">₹{sellPnL.toFixed(2)}</div>
              <div className="text-sm text-slate-400">SELL P&L</div>
            </div>
          </div>
        </div>

        {/* Monthly Performance Bar Chart */}
        <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30">
          <div className="text-lg font-semibold text-white mb-4">Monthly Performance</div>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={monthlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
              <XAxis dataKey="month" stroke="#64748b" tick={{ fill: '#94a3b8' }} />
              <YAxis stroke="#64748b" tick={{ fill: '#94a3b8' }} />
              <Tooltip
                contentStyle={{
                  background: "rgba(15, 23, 42, 0.95)",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                  borderRadius: 12,
                  backdropFilter: "blur(10px)"
                }}
                formatter={(value, name) => [name === 'pnl' ? `₹${value}` : value, name === 'pnl' ? 'P&L' : 'Trades']}
              />
              <Bar dataKey="pnl" fill="#22d3ee" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30">
        <div className="text-lg font-semibold text-white mb-4">Performance Metrics</div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-green-400 mb-2">{(winningTrades / totalTrades * 100).toFixed(1)}%</div>
            <div className="text-slate-400">Win Rate</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-400 mb-2">{totalTrades > 0 ? (totalPnL / totalTrades).toFixed(2) : 0}</div>
            <div className="text-slate-400">Profit Factor</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-400 mb-2">{totalTrades > 0 ? (totalTrades / 30).toFixed(1) : 0}</div>
            <div className="text-slate-400">Trades per Day</div>
          </div>
        </div>
      </div>
    </div>
  );
}
