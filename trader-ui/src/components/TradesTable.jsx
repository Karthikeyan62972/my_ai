import dayjs from "dayjs";

export default function TradesTable({ rows }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30 h-[400px] flex items-center justify-center">
        <div className="text-center text-slate-400">
          <div className="text-4xl mb-2">💼</div>
          <div className="text-sm">No trades available</div>
        </div>
      </div>
    );
  }

  // Calculate summary stats
  const totalPnL = rows.reduce((sum, r) => sum + (r.pnl || 0), 0);
  const winningTrades = rows.filter(r => (r.pnl || 0) > 0).length;
  const totalTrades = rows.length;

  return (
    <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30 overflow-hidden">
      <div className="flex items-center justify-between mb-4 p-4 border-b border-slate-700/30">
        <div className="flex items-center space-x-2">
          <div className="text-lg">💼</div>
          <div className="text-lg font-semibold text-white">Trading History</div>
        </div>
        <div className="flex items-center space-x-4 text-sm">
          <div className="text-slate-400">
            Total: <span className="text-white font-medium">{totalTrades}</span>
          </div>
          <div className="text-slate-400">
            Win Rate: <span className="text-white font-medium">{totalTrades > 0 ? ((winningTrades / totalTrades) * 100).toFixed(1) : 0}%</span>
          </div>
          <div className={`font-medium ${totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            P&L: {totalPnL >= 0 ? '+' : ''}{totalPnL.toFixed(2)}
          </div>
        </div>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-slate-400 bg-slate-800/30">
            <tr>
              <th className="text-left py-3 px-4 font-medium">ID</th>
              <th className="text-left py-3 px-4 font-medium">Symbol</th>
              <th className="text-center py-3 px-4 font-medium">Side</th>
              <th className="text-right py-3 px-4 font-medium">Qty</th>
              <th className="text-right py-3 px-4 font-medium">Entry</th>
              <th className="text-right py-3 px-4 font-medium">Exit</th>
              <th className="text-right py-3 px-4 font-medium">P&L</th>
              <th className="text-center py-3 px-4 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, index) => (
              <tr 
                key={r.id} 
                className={`border-t border-slate-700/30 hover:bg-slate-800/30 transition-colors duration-200 ${
                  index % 2 === 0 ? 'bg-slate-800/10' : ''
                }`}
              >
                <td className="py-3 px-4 font-mono text-slate-300 text-sm">
                  #{r.id}
                </td>
                <td className="py-3 px-4 font-medium text-white">
                  {r.symbol}
                </td>
                <td className="text-center py-3 px-4">
                  <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                    r.side === 'BUY' 
                      ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                      : 'bg-red-500/20 text-red-400 border border-red-500/30'
                  }`}>
                    {r.side === 'BUY' ? '↗' : '↘'} {r.side}
                  </span>
                </td>
                <td className="text-right py-3 px-4 text-white font-mono">
                  {r.qty?.toLocaleString()}
                </td>
                <td className="text-right py-3 px-4 text-white font-mono">
                  ₹{r.entry_price?.toFixed(2)}
                </td>
                <td className="text-right py-3 px-4 text-slate-400 font-mono">
                  {r.exit_price ? `₹${r.exit_price.toFixed(2)}` : "—"}
                </td>
                <td className={`text-right py-3 px-4 font-mono font-medium ${
                  r.pnl > 0 
                    ? "text-green-400" 
                    : r.pnl < 0 
                    ? "text-red-400" 
                    : "text-slate-400"
                }`}>
                  {r.pnl ? (r.pnl > 0 ? '+' : '') + r.pnl.toFixed(2) : "—"}
                </td>
                <td className="text-center py-3 px-4">
                  <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                    r.status === 'CLOSED' 
                      ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                      : r.status === 'OPEN'
                      ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                      : 'bg-slate-500/20 text-slate-400 border border-slate-500/30'
                  }`}>
                    {r.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

