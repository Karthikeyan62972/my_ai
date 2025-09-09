import { useState, useEffect } from 'react';
import { fetchAllMetrics } from '../lib/api';
import SymbolPicker from './SymbolPicker';
import KPI from './KPI';
import StatusPill from './StatusPill';
import LoadingSpinner from './LoadingSpinner';
import PredictionsTable from './PredictionsTable';
import PerformanceChart from './PerformanceChart';
import VolumeChart from './VolumeChart';

export default function Dashboard() {
  const [data, setData] = useState({
    symbols: [],
    last: null,
    ticks: [],
    predictions: [],
    ticksTotal: 0,
    predictionsTotal: 0,
    ingestionStatus: null
  });
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadData = async (symbol = selectedSymbol) => {
    try {
      setLoading(true);
      setError(null);
      const metrics = await fetchAllMetrics(symbol);
      setData(metrics);
    } catch (err) {
      setError(err.message);
      console.error('Failed to load data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    if (selectedSymbol) {
      loadData(selectedSymbol);
    }
  }, [selectedSymbol]);

  // Auto-refresh every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      loadData(selectedSymbol);
    }, 5000);
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  if (loading && !data.symbols.length) {
    return <LoadingSpinner />;
  }

  // Calculate prediction metrics
  const totalPredictions = data.predictions.length;
  const hitPredictions = data.predictions.filter(p => p.hit === 1).length;
  const failedPredictions = data.predictions.filter(p => p.hit === 0).length;
  const monitoringPredictions = data.predictions.filter(p => p.status === 'MONITORING').length;
  const expiredPredictions = data.predictions.filter(p => p.status === 'EXPIRED').length;
  const hitRate = totalPredictions > 0 ? (hitPredictions / totalPredictions * 100) : 0;

  // Calculate recent performance (last 24 hours)
  const now = Date.now();
  const dayAgo = now - (24 * 60 * 60 * 1000);
  const recentPredictions = data.predictions.filter(p => p.t >= dayAgo);
  const recentHitRate = recentPredictions.length > 0 ? 
    (recentPredictions.filter(p => p.hit === 1).length / recentPredictions.length * 100) : 0;

  // Calculate average prediction accuracy by direction
  const upPredictions = data.predictions.filter(p => p.direction === 'UP');
  const downPredictions = data.predictions.filter(p => p.direction === 'DOWN');
  const upHitRate = upPredictions.length > 0 ? (upPredictions.filter(p => p.hit === 1).length / upPredictions.length * 100) : 0;
  const downHitRate = downPredictions.length > 0 ? (downPredictions.filter(p => p.hit === 1).length / downPredictions.length * 100) : 0;

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              📈 ML Trading Dashboard
            </h1>
            <p className="text-slate-400 mt-1">Real-time market analysis and prediction monitoring</p>
          </div>
          <div className="flex items-center space-x-4">
            {data.ingestionStatus && (
              <StatusPill 
                status={data.ingestionStatus.stale ? 'error' : 'success'} 
                text={data.ingestionStatus.stale ? 'Data Stale' : 'Live Data'}
              />
            )}
            <div className="text-sm text-slate-400">
              Last updated: {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
            <div className="text-red-400 font-medium">Error loading data</div>
            <div className="text-red-300 text-sm mt-1">{error}</div>
          </div>
        )}

        {/* Symbol Selection and Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <SymbolPicker 
            symbols={data.symbols} 
            value={selectedSymbol} 
            onChange={setSelectedSymbol}
          />
          <KPI 
            label="Current Price" 
            value={data.last ? `₹${data.last.ltp?.toFixed(2)}` : "—"}
            icon="💰"
          />
          <KPI 
            label="Volume" 
            value={data.last ? data.last.v?.toLocaleString() : "—"}
            icon="📊"
          />
          <KPI 
            label="Hit Rate" 
            value={`${hitRate.toFixed(1)}%`}
            icon="🎯"
          />
          <KPI 
            label="Total Predictions" 
            value={data.predictionsTotal.toLocaleString()}
            icon="🔮"
          />
        </div>

        {/* Prediction Performance Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-green-500/20 to-green-600/20 border border-green-500/30 rounded-lg p-4">
            <div className="text-green-400 text-2xl mb-2">✅</div>
            <div className="text-2xl font-bold text-white">{hitPredictions}</div>
            <div className="text-sm text-green-300">Successful Predictions</div>
          </div>
          
          <div className="bg-gradient-to-br from-red-500/20 to-red-600/20 border border-red-500/30 rounded-lg p-4">
            <div className="text-red-400 text-2xl mb-2">❌</div>
            <div className="text-2xl font-bold text-white">{failedPredictions}</div>
            <div className="text-sm text-red-300">Failed Predictions</div>
          </div>
          
          <div className="bg-gradient-to-br from-yellow-500/20 to-yellow-600/20 border border-yellow-500/30 rounded-lg p-4">
            <div className="text-yellow-400 text-2xl mb-2">⏳</div>
            <div className="text-2xl font-bold text-white">{monitoringPredictions}</div>
            <div className="text-sm text-yellow-300">Monitoring</div>
          </div>
          
          <div className="bg-gradient-to-br from-gray-500/20 to-gray-600/20 border border-gray-500/30 rounded-lg p-4">
            <div className="text-gray-400 text-2xl mb-2">⏰</div>
            <div className="text-2xl font-bold text-white">{expiredPredictions}</div>
            <div className="text-sm text-gray-300">Expired</div>
          </div>
        </div>

        {/* Direction Performance */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/30 border border-slate-700/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Direction Performance</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-400 mb-2">{upHitRate.toFixed(1)}%</div>
                <div className="text-slate-400">UP Predictions</div>
                <div className="text-sm text-slate-500">{upPredictions.length} total</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-red-400 mb-2">{downHitRate.toFixed(1)}%</div>
                <div className="text-slate-400">DOWN Predictions</div>
                <div className="text-sm text-slate-500">{downPredictions.length} total</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/30 border border-slate-700/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Recent Performance (24h)</h3>
            <div className="text-center">
              <div className="text-4xl font-bold text-blue-400 mb-2">{recentHitRate.toFixed(1)}%</div>
              <div className="text-slate-400">Hit Rate</div>
              <div className="text-sm text-slate-500">{recentPredictions.length} predictions</div>
            </div>
          </div>
        </div>

        {/* Charts */}
        {selectedSymbol && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PerformanceChart predictions={data.predictions} />
            <VolumeChart ticks={data.ticks} />
          </div>
        )}

        {/* Predictions Table */}
        {selectedSymbol && (
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/30 border border-slate-700/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Recent Predictions - {selectedSymbol}</h3>
            <PredictionsTable rows={data.predictions.slice(0, 20)} />
          </div>
        )}

        {/* System Health */}
        <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/30 border border-slate-700/30 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">System Health</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400 mb-2">{data.ticksTotal.toLocaleString()}</div>
              <div className="text-slate-400">Total Ticks Processed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-400 mb-2">{data.predictionsTotal.toLocaleString()}</div>
              <div className="text-slate-400">Total Predictions Made</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400 mb-2">{data.symbols.length}</div>
              <div className="text-slate-400">Active Symbols</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
