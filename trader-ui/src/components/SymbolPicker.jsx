export default function SymbolPicker({ symbols, value, onChange }) {
  return (
    <div className="card bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-700/30 hover:border-slate-600/50 transition-all duration-300">
      <div className="flex items-center space-x-2 mb-3">
        <div className="text-lg">📈</div>
        <div className="text-sm font-medium text-slate-300 uppercase tracking-wider">Symbol</div>
      </div>
      
      <div className="relative">
        <select
          className="w-full rounded-xl bg-slate-800/50 border border-slate-600/50 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 transition-all duration-200 text-white font-medium appearance-none cursor-pointer hover:bg-slate-700/50"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          {symbols.map((s) => (
            <option key={s} value={s} className="bg-slate-800 text-white">
              {s}
            </option>
          ))}
        </select>
        
        <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
          <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>
      
      <div className="mt-2 text-xs text-slate-500">
        {symbols.length} symbols available
      </div>
    </div>
  );
}

