import { useEffect, useState } from "react";
import {
  fetchSymbols,
  fetchLast,
  fetchCandles,
  fetchPredictions,
  fetchTrades,
  fetchDailyMetrics,
} from "./lib/api";
import SymbolPicker from "./components/SymbolPicker";
import KPI from "./components/KPI";
import CandleChart from "./components/CandleChart";
import PredictionsTable from "./components/PredictionsTable";
import TradesTable from "./components/TradesTable";

export default function App() {
  const [symbols, setSymbols] = useState([]);
  const [selected, setSelected] = useState("");
  const [last, setLast] = useState(null);
  const [candles, setCandles] = useState([]);
  const [preds, setPreds] = useState([]);
  const [trades, setTrades] = useState([]);
  const [metrics, setMetrics] = useState(null);

  // load symbols on mount
  useEffect(() => {
    fetchSymbols().then((syms) => {
      setSymbols(syms);
      if (syms.length > 0) setSelected(syms[0]);
    });
  }, []);

  // refresh data whenever selected changes
  useEffect(() => {
    if (!selected) return;
    const load = async () => {
      const [l, c, p, t, m] = await Promise.all([
        fetchLast(selected),
        fetchCandles(selected),
        fetchPredictions(selected),
        fetchTrades(),
        fetchDailyMetrics(),
      ]);
      setLast(l);
      setCandles(c);
      setPreds(p);
      setTrades(t);
      setMetrics(m);
    };
    load();

    const id = setInterval(load, 5000); // poll every 5s
    return () => clearInterval(id);
  }, [selected]);

  return (
    <div className="min-h-screen bg-ink text-white p-6 space-y-6">
      <h1 className="text-2xl font-bold mb-2">📈 Intraday ML Trader Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <SymbolPicker symbols={symbols} value={selected} onChange={setSelected} />
        <KPI label="LTP" value={last?.ltp?.toFixed(2) || "—"} />
        <KPI label="Volume" value={last?.vol_traded_today?.toLocaleString() || "—"} />
        <KPI label="Hit Rate" value={metrics?.hit_rate_pct + "%" || "—"} />
      </div>

      <CandleChart data={candles} last={last} />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PredictionsTable rows={preds} />
        <TradesTable rows={trades} />
      </div>
    </div>
  );
}

