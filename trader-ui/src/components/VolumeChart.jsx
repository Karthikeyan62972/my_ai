import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import dayjs from "dayjs";

export default function VolumeChart({ data, last }) {
  if (!data || data.length === 0) {
    return (
      <div className="card h-[300px] flex items-center justify-center text-slate-400 bg-gradient-to-br from-slate-900/50 to-slate-800/30">
        <div className="text-center">
          <div className="text-4xl mb-2">📊</div>
          <div className="text-sm text-slate-500">No volume data</div>
        </div>
      </div>
    );
  }

  const fmt = (ts) => dayjs(ts).format("HH:mm");
  
  // Calculate total volume
  const totalVolume = data.reduce((sum, d) => sum + (d.v || 0), 0);
  const avgVolume = totalVolume / data.length;

  return (
    <div className="card h-[300px] bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="text-lg font-semibold text-white">Volume Analysis</div>
          <div className="text-sm text-slate-400">Trading volume</div>
        </div>
        
        <div className="flex items-center space-x-4 text-sm">
          <div className="text-slate-400">
            Total: <span className="text-white font-medium">{totalVolume.toLocaleString()}</span>
          </div>
          <div className="text-slate-400">
            Avg: <span className="text-white font-medium">{avgVolume.toLocaleString()}</span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
          <defs>
            <linearGradient id="volumeGradient" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.8}/>
              <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.3}/>
            </linearGradient>
          </defs>
          
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
          
          <XAxis 
            dataKey="t" 
            tickFormatter={fmt} 
            stroke="#64748b" 
            tick={{ fontSize: 10, fill: '#94a3b8' }}
            axisLine={{ stroke: '#475569', strokeWidth: 1 }}
            tickLine={{ stroke: '#475569' }}
          />
          
          <YAxis 
            stroke="#64748b" 
            tick={{ fontSize: 10, fill: '#94a3b8' }}
            axisLine={{ stroke: '#475569', strokeWidth: 1 }}
            tickLine={{ stroke: '#475569' }}
          />
          
          <Tooltip
            contentStyle={{ 
              background: "rgba(15, 23, 42, 0.95)", 
              border: "1px solid rgba(148, 163, 184, 0.2)", 
              borderRadius: 12,
              backdropFilter: "blur(10px)",
              boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)"
            }}
            labelFormatter={(v) => dayjs(v).format("HH:mm:ss")}
            formatter={(val, name) => [val?.toLocaleString(), "Volume"]}
            cursor={{ fill: 'rgba(34, 211, 238, 0.1)' }}
          />
          
          <Bar 
            dataKey="v" 
            name="Volume" 
            fill="url(#volumeGradient)"
            radius={[2, 2, 0, 0]}
            opacity={0.8}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
