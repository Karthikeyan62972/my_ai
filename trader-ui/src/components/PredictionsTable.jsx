import dayjs from "dayjs";

export default function PredictionsTable({ rows }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30 h-[400px] flex items-center justify-center">
        <div className="text-center text-slate-400">
          <div className="text-4xl mb-2">🎯</div>
          <div className="text-sm">No predictions available</div>
        </div>
      </div>
    );
  }

  return (
    <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30 overflow-hidden">
      <div className="flex items-center justify-between mb-4 p-4 border-b border-slate-700/30">
        <div className="flex items-center space-x-2">
          <div className="text-lg">🎯</div>
          <div className="text-lg font-semibold text-white">Recent Predictions</div>
        </div>
        <div className="text-sm text-slate-400 bg-slate-800/50 px-3 py-1 rounded-full">
          {rows.length} predictions
        </div>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-slate-400 bg-slate-800/30">
            <tr>
              <th className="text-left py-3 px-4 font-medium">Time</th>
              <th className="text-right py-3 px-4 font-medium">Current</th>
              <th className="text-right py-3 px-4 font-medium">Predicted</th>
              <th className="text-center py-3 px-4 font-medium">Direction</th>
              <th className="text-center py-3 px-4 font-medium">Result</th>
              <th className="text-right py-3 px-4 font-medium">Expiry</th>
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
                <td className="py-3 px-4 font-medium text-slate-300">
                  {dayjs(r.t).format("HH:mm:ss")}
                </td>
                <td className="text-right py-3 px-4 text-white font-mono">
                  {r.price_now?.toFixed(2)}
                </td>
                <td className="text-right py-3 px-4 text-white font-mono">
                  {r.pred_price?.toFixed(2)}
                </td>
                <td className="text-center py-3 px-4">
                  <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                    r.direction === 'UP' 
                      ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                      : 'bg-red-500/20 text-red-400 border border-red-500/30'
                  }`}>
                    {r.direction === 'UP' ? '↗' : '↘'} {r.direction}
                  </span>
                </td>
                <td className="text-center py-3 px-4">
                  {r.hit === 1 ? (
                    <span className="inline-flex items-center justify-center w-6 h-6 bg-green-500/20 text-green-400 rounded-full border border-green-500/30">
                      ✔
                    </span>
                  ) : r.hit === 0 ? (
                    <span className="inline-flex items-center justify-center w-6 h-6 bg-red-500/20 text-red-400 rounded-full border border-red-500/30">
                      ✖
                    </span>
                  ) : (
                    <span className="inline-flex items-center justify-center w-6 h-6 bg-slate-500/20 text-slate-400 rounded-full border border-slate-500/30">
                      —
                    </span>
                  )}
                </td>
                <td className="text-right py-3 px-4 text-slate-400 font-mono">
                  {dayjs(r.expiry).format("HH:mm:ss")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

