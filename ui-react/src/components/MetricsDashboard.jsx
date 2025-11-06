import { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  Clock,
  Zap,
  Database,
  Cpu,
  Activity,
  ChevronDown,
  ChevronUp,
  RefreshCw
} from 'lucide-react';
import { agentAPI } from '../services/api';

const MetricCard = ({ icon: Icon, label, value, subtext, color = 'cyan' }) => {
  const colorClasses = {
    cyan: 'text-cyan-400 border-cyan-600/40 bg-gradient-to-br from-cyan-900/40 to-teal-900/40',
    emerald: 'text-emerald-400 border-emerald-600/40 bg-gradient-to-br from-emerald-900/40 to-green-900/40',
    orange: 'text-orange-400 border-orange-600/40 bg-gradient-to-br from-orange-900/40 to-amber-900/40',
    purple: 'text-purple-400 border-purple-600/40 bg-gradient-to-br from-purple-900/40 to-indigo-900/40',
  };

  return (
    <div className={`${colorClasses[color]} border-2 rounded-lg p-4 backdrop-blur-sm`}>
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-slate-700/50 rounded flex items-center justify-center border border-slate-600">
          <Icon size={20} className={colorClasses[color].split(' ')[0]} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-xs text-slate-400 uppercase tracking-wide">{label}</div>
          <div className="text-2xl font-bold text-slate-100">{value}</div>
          {subtext && <div className="text-xs text-slate-500 mt-0.5">{subtext}</div>}
        </div>
      </div>
    </div>
  );
};

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);

  const loadMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await agentAPI.getMetrics();
      if (data) {
        setMetrics(data);
        setLastUpdate(new Date());
      } else {
        setError('No metrics data available');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Initial load
    loadMetrics();

    // Auto-refresh every 30 seconds
    const interval = setInterval(loadMetrics, 30000);

    return () => clearInterval(interval);
  }, []);

  const formatDuration = (seconds) => {
    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    return `${(seconds / 60).toFixed(1)}m`;
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  if (loading && !metrics) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="bg-slate-800/50 rounded-lg border border-slate-700/50 p-4">
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <RefreshCw size={16} className="animate-spin" />
            <span>Loading metrics...</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="bg-orange-900/20 border border-orange-600/30 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-orange-400">
              <Activity size={16} />
              <span>Failed to load metrics: {error}</span>
            </div>
            <button
              onClick={loadMetrics}
              className="px-3 py-1 bg-orange-600/20 hover:bg-orange-600/30 border border-orange-600/40 rounded text-xs text-orange-300 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!metrics) return null;

  const { runs: run_metrics, execution: execution_metrics, tools: tool_metrics, planner: planner_metrics } = metrics;

  return (
    <div className="max-w-7xl mx-auto px-6 py-4">
      <div className="bg-slate-800/50 rounded-lg border border-slate-700/50 p-6 backdrop-blur-sm">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-slate-100 uppercase tracking-wide flex items-center gap-2">
            <BarChart3 size={20} className="text-cyan-400" />
            System Metrics
          </h3>
          <div className="flex items-center gap-3">
            {lastUpdate && (
              <span className="text-xs text-slate-500">
                Updated {((new Date() - lastUpdate) / 1000).toFixed(0)}s ago
              </span>
            )}
            <button
              onClick={loadMetrics}
              className="p-1.5 hover:bg-slate-700/50 rounded transition-colors"
              title="Refresh metrics"
            >
              <RefreshCw size={16} className={`text-slate-400 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Primary Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <MetricCard
            icon={Activity}
            label="Total Runs"
            value={run_metrics?.total || 0}
            subtext={`${run_metrics?.by_status?.completed || 0} completed`}
            color="cyan"
          />

          <MetricCard
            icon={TrendingUp}
            label="Success Rate"
            value={formatPercentage(run_metrics?.success_rate || 0)}
            subtext={`${run_metrics?.by_status?.failed || 0} failed`}
            color="emerald"
          />

          <MetricCard
            icon={Clock}
            label="Avg Duration"
            value={formatDuration(execution_metrics?.avg_duration_seconds || 0)}
            subtext={`${execution_metrics?.total_steps_executed || 0} steps total`}
            color="purple"
          />

          <MetricCard
            icon={Zap}
            label="Tool Executions"
            value={tool_metrics?.total_executions || 0}
            subtext={`${Object.keys(tool_metrics?.by_tool || {}).length} tools used`}
            color="orange"
          />
        </div>

        {/* Advanced Metrics (Collapsible) */}
        <details
          className="bg-slate-900/30 rounded-lg border border-slate-700/30"
          open={showAdvanced}
          onToggle={(e) => setShowAdvanced(e.target.open)}
        >
          <summary className="cursor-pointer px-4 py-3 font-semibold text-sm text-cyan-400 hover:text-cyan-300 uppercase tracking-wide flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Cpu size={16} />
              Advanced Metrics
            </span>
            {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </summary>

          <div className="px-4 pb-4 pt-2">
            {!planner_metrics && (!tool_metrics?.by_tool || Object.keys(tool_metrics.by_tool).length === 0) ? (
              <div className="text-center py-8">
                <div className="text-slate-500 text-sm">
                  <Cpu size={32} className="mx-auto mb-2 opacity-50" />
                  <p>No advanced metrics available yet.</p>
                  <p className="text-xs mt-1">Run some tasks to see planner performance and tool usage.</p>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* Planner Performance */}
                {planner_metrics && (
                <>
                  <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Database size={14} className="text-purple-400" />
                      <span className="text-xs text-slate-400 uppercase tracking-wide">Planner Type</span>
                    </div>
                    <div className="text-lg font-bold text-slate-100">
                      {planner_metrics.llm_plans || 0} LLM / {planner_metrics.pattern_plans || 0} Pattern
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      {planner_metrics.cached_plans || 0} cached
                    </div>
                  </div>

                  <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity size={14} className="text-orange-400" />
                      <span className="text-xs text-slate-400 uppercase tracking-wide">Cache Performance</span>
                    </div>
                    <div className="text-lg font-bold text-slate-100">
                      {formatPercentage(planner_metrics.cache_hit_rate || 0)}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      hit rate
                    </div>
                  </div>

                  <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Clock size={14} className="text-cyan-400" />
                      <span className="text-xs text-slate-400 uppercase tracking-wide">Planning Latency</span>
                    </div>
                    <div className="text-lg font-bold text-slate-100">
                      {planner_metrics.avg_latency_ms?.toFixed(0) || 0}ms
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      average
                    </div>
                  </div>

                  <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingUp size={14} className="text-emerald-400" />
                      <span className="text-xs text-slate-400 uppercase tracking-wide">LLM Tokens</span>
                    </div>
                    <div className="text-lg font-bold text-slate-100">
                      {planner_metrics.avg_tokens_per_plan?.toFixed(0) || 0}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      per plan
                    </div>
                  </div>

                  <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity size={14} className="text-orange-400" />
                      <span className="text-xs text-slate-400 uppercase tracking-wide">Fallback Rate</span>
                    </div>
                    <div className="text-lg font-bold text-slate-100">
                      {formatPercentage(planner_metrics.fallback_rate || 0)}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      LLM failures
                    </div>
                  </div>
                </>
              )}

              {/* Tool Usage Breakdown */}
              {tool_metrics?.by_tool && (
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap size={14} className="text-cyan-400" />
                    <span className="text-xs text-slate-400 uppercase tracking-wide">Tool Breakdown</span>
                  </div>
                  <div className="space-y-1">
                    {Object.entries(tool_metrics.by_tool).map(([tool, count]) => (
                      <div key={tool} className="flex justify-between text-sm">
                        <span className="text-slate-400">{tool}</span>
                        <span className="text-slate-100 font-mono">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              </div>
            )}
          </div>
        </details>
      </div>
    </div>
  );
}
