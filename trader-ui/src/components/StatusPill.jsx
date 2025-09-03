export default function StatusPill({ status }) {
  if (!status) {
    return (
      <div className="px-3 py-1 rounded-full bg-slate-500/20 text-slate-400 border border-slate-500/30 text-sm">
        Unknown
      </div>
    );
  }

  const isStale = status.stale || false;
  const isActive = status.active || false;

  let colorClasses = "bg-slate-500/20 text-slate-400 border-slate-500/30";
  let text = "Unknown";

  if (isStale) {
    colorClasses = "bg-red-500/20 text-red-400 border-red-500/30";
    text = "Stale";
  } else if (isActive) {
    colorClasses = "bg-green-500/20 text-green-400 border-green-500/30";
    text = "Active";
  } else {
    colorClasses = "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
    text = "Inactive";
  }

  return (
    <div className={`px-3 py-1 rounded-full ${colorClasses} border text-sm font-medium flex items-center gap-2`}>
      <div className={`w-2 h-2 rounded-full ${isStale ? 'bg-red-400' : isActive ? 'bg-green-400' : 'bg-yellow-400'} animate-pulse`}></div>
      {text}
    </div>
  );
}
