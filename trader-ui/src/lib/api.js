import axios from "axios";

export const API = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || "http://localhost:8000",
  timeout: 10000,
});

export const fetchSymbols = () => API.get("/api/symbols").then(r => r.data);
export const fetchLast = (symbol) => API.get("/api/last", { params: { symbol } }).then(r => r.data);
export const fetchCandles = (symbol, interval="1m", lookback=180) =>
  API.get("/api/candles", { params: { symbol, interval, lookback } }).then(r => r.data);
export const fetchPredictions = (symbol, limit=200) =>
  API.get("/api/predictions", { params: { symbol, limit } }).then(r => r.data);
export const fetchTrades = (status) =>
  API.get("/api/trades", { params: status ? { status } : {} }).then(r => r.data);
export const fetchDailyMetrics = (date) =>
  API.get("/api/metrics/daily", { params: date ? { date } : {} }).then(r => r.data);

export const fetchIngestionStatus = (symbol, threshold = 30) =>
  API.get("/api/ingestion/status", { params: { symbol, threshold } }).then(r => r.data);

export const fetchIngestionSummary = (threshold = 30) =>
  API.get("/api/ingestion/summary", { params: { threshold } }).then(r => r.data);


