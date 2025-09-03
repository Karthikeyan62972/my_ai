import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Area, AreaChart } from "recharts";
import dayjs from "dayjs";

export default function PerformanceChart({ trades }) {
  if (!trades || trades.length === 0) {
    return (
      <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30 h-[300px] flex items-center justify-center">
        <div className="text-center text-slate-400">
          <div className="text-4xl mb-2">📊</div>
          <div className="text-sm">No performance data</div>
        </div>
      </div>
    );
  }

  // Process trades data for performance chart
  const performanceData = trades
    .filter(t => t.pnl !== null && t.pnl !== undefined)
    .map(t => ({
      date: dayjs(t.exit_time || t.entry_time).format('MMM DD'),
      pnl: t.pnl || 0,
      cumulative: 0 // Will be calculated below
    }))
    .sort((a, b) => dayjs(a.date, 'MMM DD').diff(dayjs(b.date, 'MMM DD')));

  // Calculate cumulative P&L
  let cumulative = 0;
  performanceData.forEach(item => {
    cumulative += item.pnl;
    item.cumulative = cumulative;
  });

  // Calculate summary stats
  const totalPnL = performanceData.reduce((sum, d) => sum + d.pnl, 0);
  const maxDrawdown = Math.min(...performanceData.map(d => d.cumulative));
  const maxProfit = Math.max(...performanceData.map(d => d.cumulative));

  return (
    <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="text-lg">📊</div>
          <div className="text-lg font-semibold text-white">Performance Overview</div>
        </div>
        
        <div className="flex items-center space-x-4 text-sm">
          <div className="text-slate-400">
            Total: <span className={`font-medium ${totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {totalPnL >= 0 ? '+' : ''}{totalPnL.toFixed(2)}
            </span>
          </div>
          <div className="text-slate-400">
            Max Profit: <span className="text-green-400 font-medium">+{maxProfit.toFixed(2)}</span>
          </div>
          <div className="text-slate-400">
            Max Drawdown: <span className="text-red-400 font-medium">{maxDrawdown.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={250}>
        <AreaChart data={performanceData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
          <defs>
            <linearGradient id="performanceGradient" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.8}/>
              <stop offset="50%" stopColor="#22d3ee" stopOpacity={0.3}/>
              <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.05}/>
            </linearGradient>
            <linearGradient id="lineGradient" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stopColor="#22d3ee"/>
              <stop offset="100%" stopColor="#3b82f6"/>
            </linearGradient>
          </defs>
          
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
          
          <XAxis 
            dataKey="date" 
            stroke="#64748b" 
            tick={{ fontSize: 11, fill: '#94a3b8' }}
            axisLine={{ stroke: '#475569', strokeWidth: 1 }}
            tickLine={{ stroke: '#475569' }}
          />
          
          <YAxis 
            stroke="#64748b" 
            tick={{ fontSize: 11, fill: '#94a3b8' }}
            axisLine={{ stroke: '#475569', strokeWidth: 1 }}
            tickLine={{ stroke: '#475569' }}
            tickFormatter={(value) => value.toFixed(0)}
          />
          
          <Tooltip
            contentStyle={{ 
              background: "rgba(15, 23, 42, 0.95)", 
              border: "1px solid rgba(148, 163, 184, 0.2)", 
              borderRadius: 12,
              backdropFilter: "blur(10px)",
              boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)"
            }}
            formatter={(val, name) => [
              name === 'cumulative' ? `₹${val.toFixed(2)}` : `₹${val.toFixed(2)}`, 
              name === 'cumulative' ? 'Cumulative P&L' : 'Trade P&L'
            ]}
            cursor={{ stroke: '#22d3ee', strokeWidth: 1, strokeDasharray: '3 3' }}
          />
          
          <Area 
            dataKey="cumulative" 
            name="Cumulative P&L" 
            type="monotone" 
            stroke="url(#lineGradient)" 
            strokeWidth={2}
            fill="url(#performanceGradient)" 
            dot={false}
            activeDot={{ r: 4, fill: '#22d3ee', stroke: '#0f172a', strokeWidth: 2 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
