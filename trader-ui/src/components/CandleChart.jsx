import dayjs from "dayjs";

export default function StatusPill({ status }) {
  if (!status) {
    return (
      <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-700/50 text-slate-300 text-xs">
        <span className="w-2 h-2 rounded-full bg-slate-400 animate-pulse" />
        checking…
      </span>
    );
  }
  const stale = status.stale;
  const age = status.age_sec ?? 0;
  const color = stale ? "bg-rose-500" : "bg-emerald-500";
  const text = stale ? `STALE ${age}s` : "LIVE";

  return (
    <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-card/80 border border-white/10 text-xs">
      <span className={`w-2 h-2 rounded-full ${color} animate-pulse`} />
      <span className={stale ? "text-rose-300" : "text-emerald-300"}>{text}</span>
      {status.last_ts && (
        <span className="text-slate-400">• last {dayjs(status.last_ts).format("HH:mm:ss")}</span>
      )}
    </span>
  );
}
