import React, { useState, useEffect } from 'react';
import { PlayCircle, CheckCircle, XCircle, Clock, Loader, Activity, AlertCircle, TrendingUp } from 'lucide-react';
import { agentAPI, pollRunStatus } from '../services/api';
import MetricsDashboard from './MetricsDashboard';

const StatusBadge = ({ status }) => {
  const configs = {
    pending: { icon: Clock, color: "bg-slate-700 text-slate-300", label: "Pending" },
    running: { icon: Loader, color: "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30", label: "Running" },
    completed: { icon: CheckCircle, color: "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30", label: "Completed" },
    failed: { icon: XCircle, color: "bg-orange-500/20 text-orange-400 border border-orange-500/30", label: "Failed" }
  };
  
  const config = configs[status] || configs.pending;
  const Icon = config.icon;
  
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded text-xs font-semibold uppercase tracking-wider ${config.color}`}>
      <Icon size={12} className={status === 'running' ? 'animate-spin' : ''} />
      {config.label}
    </span>
  );
};

const ThoughtBubble = ({ children, type = "thinking" }) => {
  const colors = {
    thinking: "bg-slate-800/50 border-slate-700 text-slate-200",
    action: "bg-teal-900/30 border-teal-700/50 text-teal-100",
    result: "bg-emerald-900/30 border-emerald-700/50 text-emerald-100",
    error: "bg-orange-900/30 border-orange-700/50 text-orange-100"
  };
  
  return (
    <div className={`relative rounded-lg border p-4 backdrop-blur-sm ${colors[type]}`}>
      <div className={`absolute -left-2 top-6 w-4 h-4 rotate-45 border-l border-b ${colors[type].split('border-')[1]?.split(' ')[0] ? 'border-' + colors[type].split('border-')[1].split(' ')[0] : ''} ${colors[type].split(' ')[0]}`}></div>
      {children}
    </div>
  );
};

const TodoItem = ({ todo, isNew = false }) => {
  // Format timestamp to relative or absolute time
  const formatTimestamp = (isoString) => {
    if (!isoString) return '';

    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    // For older items, show the date
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
    });
  };

  return (
    <div className={`flex items-start gap-3 p-3 rounded-lg border transition-all ${
      isNew
        ? 'bg-emerald-900/20 border-emerald-600/40 shadow-lg shadow-emerald-500/10'
        : 'bg-slate-800/50 border-slate-700/50 hover:border-slate-600'
    }`}>
      <div className={`flex-shrink-0 w-5 h-5 rounded border-2 mt-0.5 transition-all ${
        todo.completed
          ? 'bg-emerald-500 border-emerald-500 flex items-center justify-center'
          : 'border-slate-600 bg-slate-900/50'
      }`}>
        {todo.completed && (
          <CheckCircle size={14} className="text-slate-900" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className={`text-sm font-medium ${
          todo.completed ? 'line-through text-slate-500' : 'text-slate-200'
        }`}>
          {todo.text}
        </div>
        <div className="flex items-center gap-2 mt-1 flex-wrap">
          <span className="text-xs font-mono text-slate-500">{todo.id}</span>
          {todo.created_at && (
            <span className="text-xs text-slate-400 flex items-center gap-1">
              <Clock size={12} className="text-slate-500" />
              {formatTimestamp(todo.created_at)}
            </span>
          )}
          {isNew && (
            <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded font-semibold border border-emerald-500/30">
              NEW
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

const TodoListRenderer = ({ output, toolName }) => {
  // Handle add action (single todo object)
  // Backend returns {todo: {...}} for add action
  if ((toolName === 'TodoStore.add' || toolName === 'todo_store') && (output.id || output.todo)) {
    const todo = output.todo || output;
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm text-emerald-400 font-semibold">
          <CheckCircle size={16} />
          <span className="uppercase tracking-wide">Task Created</span>
        </div>
        <TodoItem todo={todo} isNew={true} />
      </div>
    );
  }

  // Handle list action (array of todos)
  // Backend returns array directly, not wrapped in {todos: [...]}
  const todosList = Array.isArray(output) ? output : output?.todos;

  if ((toolName === 'TodoStore.list' || toolName === 'todo_store') && todosList) {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between px-3 py-2 bg-slate-800/30 rounded-lg border border-slate-700/50">
          <div className="text-sm font-semibold text-cyan-400 uppercase tracking-wide">
            {todosList.length} {todosList.length === 1 ? 'Task' : 'Tasks'}
          </div>
          <div className="text-xs text-slate-400">
            {todosList.filter(t => t.completed).length} completed
          </div>
        </div>
        {todosList.length > 0 ? (
          <div className="space-y-2">
            {todosList.map((todo) => (
              <TodoItem key={todo.id} todo={todo} />
            ))}
          </div>
        ) : (
          <div className="text-sm text-slate-500 italic p-6 bg-slate-800/30 rounded-lg border-2 border-dashed border-slate-700/50 text-center">
            No tasks found
          </div>
        )}
      </div>
    );
  }
  
  if ((toolName === 'TodoStore.complete' || toolName === 'todo_store') && (output.id || output.todo)) {
    const todo = output.todo || output;
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm text-emerald-400 font-semibold">
          <CheckCircle size={16} />
          <span className="uppercase tracking-wide">Task Completed</span>
        </div>
        <TodoItem todo={todo} />
      </div>
    );
  }

  if ((toolName === 'TodoStore.delete' || toolName === 'todo_store') && (output.id || output.deleted_todo)) {
    return (
      <div className="flex items-center gap-2 text-sm text-emerald-400 p-3 bg-emerald-900/20 rounded-lg border border-emerald-600/30 font-semibold">
        <CheckCircle size={16} />
        <span className="uppercase tracking-wide">Task Deleted</span>
      </div>
    );
  }
  
  return null;
};

const CalculatorRenderer = ({ output }) => {
  // Calculator returns the result as a direct number, not wrapped in {result: number}
  if (output !== null && output !== undefined && typeof output === 'number') {
    // Format number: remove unnecessary decimals, add commas for readability
    const formattedNumber = output % 1 === 0
      ? output.toLocaleString() // Integer: add commas
      : output.toLocaleString(undefined, { maximumFractionDigits: 6 }); // Decimal: limit to 6 places

    return (
      <div className="bg-gradient-to-br from-cyan-900/40 to-teal-900/40 border-2 border-cyan-600/40 rounded-lg p-2 text-center backdrop-blur-sm">
        <div className="text-xs font-semibold text-cyan-400 uppercase tracking-wide mb-1">
          Result
        </div>
        <div className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 to-teal-300 font-mono">
          {formattedNumber}
        </div>
      </div>
    );
  }
  return null;
};

const OutputRenderer = ({ output, toolName }) => {
  if (!toolName) {
    return (
      <div className="text-sm">
        <pre className="font-mono text-slate-300 whitespace-pre-wrap break-all">
          {typeof output === 'object' ? JSON.stringify(output, null, 2) : String(output)}
        </pre>
      </div>
    );
  }

  if (toolName.startsWith('TodoStore.') || toolName === 'todo_store') {
    const rendered = <TodoListRenderer output={output} toolName={toolName} />;
    if (rendered) return rendered;
  }

  if (toolName.startsWith('Calculator.') || toolName === 'calculator') {
    const rendered = <CalculatorRenderer output={output} />;
    if (rendered) return rendered;
  }
  
  return (
    <div className="text-sm">
      {typeof output === 'object' ? (
        <div className="space-y-1">
          {Object.entries(output).map(([key, value]) => (
            <div key={key} className="text-slate-300">
              <span className="font-medium text-cyan-400">{key}:</span>{' '}
              <span className="font-mono text-slate-400">
                {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <span className="font-mono text-slate-300">{String(output)}</span>
      )}
    </div>
  );
};

const StepFlow = ({ step, execution, isLast }) => {
  const status = execution?.status || 'pending';
  const hasError = execution?.error;
  
  return (
    <div className="relative">
      {!isLast && (
        <div className="absolute left-6 top-16 bottom-0 w-0.5 bg-gradient-to-b from-cyan-700/50 to-teal-700/30 z-0"></div>
      )}
      
      <div className="relative z-10">
        <div className="flex items-start gap-4 mb-4">
          <div className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg shadow-lg transition-all border-2 ${
            status === 'completed' ? 'bg-gradient-to-br from-emerald-600 to-emerald-700 border-emerald-500/50 text-white shadow-emerald-500/20' :
            status === 'failed' ? 'bg-gradient-to-br from-orange-600 to-orange-700 border-orange-500/50 text-white shadow-orange-500/20' :
            status === 'running' ? 'bg-gradient-to-br from-cyan-600 to-cyan-700 border-cyan-500/50 text-white animate-pulse shadow-cyan-500/20' :
            'bg-slate-700 border-slate-600 text-slate-400'
          }`}>
            {status === 'completed' ? <CheckCircle size={24} /> :
             status === 'failed' ? <XCircle size={24} /> :
             status === 'running' ? <Loader size={24} className="animate-spin" /> :
             step.step_number}
          </div>
          
          <div className="flex-1 pt-2">
            <div className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">
              Step {step.step_number}
            </div>
            <div className="text-sm text-cyan-400 font-mono font-semibold">{step.tool_name}</div>
          </div>
        </div>
        
        <div className="ml-16 space-y-4">
          <ThoughtBubble type="thinking">
            <div className="flex items-start gap-2">
              <Activity size={16} className="text-slate-400 mt-0.5 flex-shrink-0" />
              <div>
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Analysis</div>
                <div className="text-sm leading-relaxed">{step.reasoning}</div>
              </div>
            </div>
          </ThoughtBubble>
          
          <ThoughtBubble type="action">
            <div className="flex items-start gap-2">
              <PlayCircle size={16} className="text-teal-400 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <div className="text-xs font-bold text-teal-400 uppercase tracking-wider mb-2">Execution</div>
                <div className="text-sm space-y-1">
                  {Object.entries(step.tool_input || {}).map(([key, value]) => (
                    <div key={key} className="text-teal-100">
                      <span className="font-semibold text-cyan-300">{key}:</span>{' '}
                      <span className="font-mono text-slate-300">{typeof value === 'object' ? JSON.stringify(value) : value}</span>
                    </div>
                  ))}
                  {Object.keys(step.tool_input || {}).length === 0 && (
                    <span className="text-slate-500 italic">No parameters required</span>
                  )}
                </div>
              </div>
            </div>
          </ThoughtBubble>
          
          {execution && (
            <>
              {hasError ? (
                <ThoughtBubble type="error">
                  <div className="flex items-start gap-2">
                    <AlertCircle size={16} className="text-orange-400 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <div className="text-xs font-bold text-orange-400 uppercase tracking-wider mb-1">Error</div>
                      <div className="text-sm mb-2">{execution.error}</div>
                      {execution.retry_count && (
                        <div className="text-xs bg-orange-500/20 border border-orange-500/30 px-2 py-1 rounded inline-block text-orange-300 font-semibold uppercase tracking-wide">
                          Retried {execution.retry_count + 1}x
                        </div>
                      )}
                    </div>
                  </div>
                </ThoughtBubble>
              ) : execution.output ? (
                <ThoughtBubble type="result">
                  <div className="flex items-start gap-2">
                    <CheckCircle size={16} className="text-emerald-400 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <div className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-3">Result</div>
                      <OutputRenderer output={execution.output} toolName={step.tool_name} />
                      {execution.duration_ms !== undefined && execution.duration_ms !== null && (
                        <div className="text-xs text-emerald-400/70 mt-3 font-mono">
                          ⚡ {execution.duration_ms < 1
                              ? `${(execution.duration_ms * 1000).toFixed(2)}µs`
                              : execution.duration_ms < 1000
                              ? `${execution.duration_ms.toFixed(2)}ms`
                              : `${(execution.duration_ms / 1000).toFixed(2)}s`}
                        </div>
                      )}
                    </div>
                  </div>
                </ThoughtBubble>
              ) : null}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

const RunDetails = ({ run }) => {
  if (!run) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-md">
          <div className="mb-6 relative">
            <div className="w-24 h-24 mx-auto bg-gradient-to-br from-cyan-900/50 to-teal-900/50 rounded-full flex items-center justify-center border-2 border-cyan-700/50">
              <Activity size={48} className="text-cyan-400" />
            </div>
            <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 bg-slate-800 px-3 py-1 rounded-full border-2 border-cyan-600/50 text-cyan-400 font-bold text-sm uppercase tracking-wide">
              Ready
            </div>
          </div>
          <h3 className="text-2xl font-bold text-slate-200 mb-3">Agent Execution Monitor</h3>
          <p className="text-slate-400 mb-6 leading-relaxed">
            Create a new execution or select from history to track the agent's planning and execution process in real-time.
          </p>
          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 text-left text-sm text-slate-300">
            <div className="font-bold mb-2 text-cyan-400 uppercase tracking-wide">Quick Start</div>
            <ul className="space-y-1 text-slate-400">
              <li>→ Add and list todos</li>
              <li>→ Perform calculations</li>
              <li>→ Chain multiple operations</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }
  
  const duration = run.completed_at ? 
    new Date(run.completed_at) - new Date(run.created_at) : 
    null;
  
  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-slate-100 uppercase tracking-wide">Execution Path</h2>
          <StatusBadge status={run.status} />
        </div>
        
        <div className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 border-2 border-cyan-700/40 rounded-lg p-6 mb-6 relative overflow-hidden backdrop-blur-sm">
          <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-600/10 rounded-full -mr-16 -mt-16"></div>
          <div className="absolute bottom-0 left-0 w-24 h-24 bg-teal-600/10 rounded-full -ml-12 -mb-12"></div>
          <div className="relative">
            <div className="flex items-start gap-3 mb-4">
              <TrendingUp size={20} className="text-cyan-400 mt-1 flex-shrink-0" />
              <div className="flex-1">
                <div className="text-xs font-bold text-cyan-400 mb-2 uppercase tracking-widest">User Request</div>
                <div className="text-lg text-slate-100 leading-relaxed font-medium">{run.prompt}</div>
              </div>
            </div>

            {/* Processing indicator for placeholder state */}
            {run.isPlaceholder && (
              <div className="mt-4 pt-4 border-t border-slate-700/50">
                <div className="flex items-center gap-3 text-cyan-400">
                  <Loader size={18} className="animate-spin" />
                  <div className="text-sm font-semibold">
                    Planning execution strategy...
                  </div>
                </div>
              </div>
            )}

            {!run.isPlaceholder && run.plan && (
              <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-700/50">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-slate-700/50 rounded flex items-center justify-center border border-slate-600">
                    <span className="text-lg font-bold text-cyan-400">{run.plan.steps.length}</span>
                  </div>
                  <div className="text-sm">
                    <div className="font-bold text-slate-300 uppercase tracking-wide">Steps</div>
                    <div className="text-xs text-slate-500">in execution plan</div>
                  </div>
                </div>
              
                {duration && (
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-slate-700/50 rounded flex items-center justify-center border border-slate-600">
                      <Clock size={18} className="text-cyan-400" />
                    </div>
                    <div className="text-sm">
                      <div className="font-bold text-slate-300">{duration}ms</div>
                      <div className="text-xs text-slate-500 uppercase tracking-wide">Duration</div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Performance Summary */}
        {run.execution_log && run.execution_log.length > 0 && (() => {
          const totalExecutionTime = run.execution_log.reduce((sum, step) => sum + (step.duration_ms || 0), 0);

          // Don't show performance summary if no timing data is available (old runs)
          if (totalExecutionTime === 0) {
            return null;
          }

          const avgTimePerStep = totalExecutionTime / run.execution_log.length;
          const slowestStep = run.execution_log.reduce((max, step) =>
            (step.duration_ms || 0) > (max.duration_ms || 0) ? step : max, run.execution_log[0]
          );
          const totalRetries = run.execution_log.reduce((sum, step) => sum + ((step.attempts || 1) - 1), 0);
          const toolsUsed = [...new Set(run.execution_log.map(step => step.tool_name).filter(Boolean))];

          return (
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-4 mb-4">
              <h3 className="font-semibold text-emerald-400 uppercase tracking-wide text-sm mb-3">
                Performance Summary
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                <div>
                  <div className="text-xs text-slate-500 uppercase tracking-wide">Total Execution</div>
                  <div className="text-lg font-bold text-slate-100">{totalExecutionTime.toFixed(0)}ms</div>
                </div>
                <div>
                  <div className="text-xs text-slate-500 uppercase tracking-wide">Avg per Step</div>
                  <div className="text-lg font-bold text-slate-100">{avgTimePerStep.toFixed(0)}ms</div>
                </div>
                <div>
                  <div className="text-xs text-slate-500 uppercase tracking-wide">Slowest Step</div>
                  <div className="text-lg font-bold text-slate-100">
                    #{slowestStep.step_number} ({(slowestStep.duration_ms || 0).toFixed(0)}ms)
                  </div>
                </div>
                {totalRetries > 0 && (
                  <div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide">Total Retries</div>
                    <div className="text-lg font-bold text-orange-400">{totalRetries}</div>
                  </div>
                )}
                <div className="md:col-span-2">
                  <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Tools Used</div>
                  <div className="flex flex-wrap gap-1">
                    {toolsUsed.map(tool => (
                      <span key={tool} className="px-2 py-0.5 bg-cyan-900/30 border border-cyan-600/30 rounded text-xs text-cyan-300">
                        {tool}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          );
        })()}

        <details className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-4 text-sm">
          <summary className="cursor-pointer font-semibold text-cyan-400 hover:text-cyan-300 uppercase tracking-wide">
            Technical Metadata
          </summary>
          <div className="mt-3 space-y-2 text-slate-400 font-mono text-xs">
            <div>
              <span className="text-slate-500">Run ID:</span>
              <span className="ml-2 text-slate-300">{run.run_id}</span>
            </div>
            <div>
              <span className="text-slate-500">Created:</span>
              <span className="ml-2 text-slate-300">{new Date(run.created_at).toLocaleString()}</span>
            </div>
            {run.started_at && (
              <div>
                <span className="text-slate-500">Started:</span>
                <span className="ml-2 text-slate-300">{new Date(run.started_at).toLocaleString()}</span>
              </div>
            )}
            {run.completed_at && (
              <div>
                <span className="text-slate-500">Completed:</span>
                <span className="ml-2 text-slate-300">{new Date(run.completed_at).toLocaleString()}</span>
              </div>
            )}
          </div>
        </details>
      </div>
      
      {/* Show placeholder message when in placeholder state */}
      {run.isPlaceholder ? (
        <div className="bg-slate-800/30 border border-slate-700/30 rounded-lg p-8">
          <div className="flex flex-col items-center gap-4 text-center">
            <Loader size={32} className="animate-spin text-cyan-400" />
            <div>
              <div className="text-lg font-semibold text-slate-200 mb-2">
                Analyzing your request...
              </div>
              <div className="text-sm text-slate-400">
                Creating execution plan and preparing tools
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div>
          <div className="flex items-center gap-2 mb-6">
            <div className="h-0.5 w-12 bg-gradient-to-r from-cyan-600 to-teal-600 rounded-full"></div>
            <h3 className="text-lg font-bold text-slate-200 uppercase tracking-wide">Execution Timeline</h3>
          </div>
          <div className="space-y-8">
            {run.plan?.steps?.map((step, idx) => (
              <StepFlow
                key={idx}
                step={step}
                execution={run.execution_log?.find(e => e.step_number === step.step_number)}
                isLast={idx === (run.plan?.steps?.length || 1) - 1}
              />
            ))}
          </div>
        
        {run.status === 'completed' && (
          <div className="mt-8 p-4 bg-gradient-to-r from-emerald-900/30 to-teal-900/30 border-2 border-emerald-600/40 rounded-lg backdrop-blur-sm">
            <div className="flex items-center gap-2 text-emerald-400 font-bold uppercase tracking-wide">
              <CheckCircle size={20} />
              <span>Execution Completed Successfully</span>
            </div>
          </div>
        )}
        
          {run.status === 'failed' && (
            <div className="mt-8 p-4 bg-gradient-to-r from-orange-900/30 to-red-900/30 border-2 border-orange-600/40 rounded-lg backdrop-blur-sm">
              <div className="flex items-center gap-2 text-orange-400 font-bold uppercase tracking-wide">
                <XCircle size={20} />
                <span>Execution Failed - Review Error Details</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default function AgentRuntimeUI() {
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [healthStatus, setHealthStatus] = useState("checking");
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load initial data on mount
  useEffect(() => {
    loadRecentRuns();
    checkSystemHealth();
    
    // Refresh health status every 30 seconds
    const healthInterval = setInterval(checkSystemHealth, 30000);
    
    return () => clearInterval(healthInterval);
  }, []);

  const loadRecentRuns = async () => {
    try {
      setIsLoading(true);
      const recentRuns = await agentAPI.getRecentRuns(10);
      setRuns(recentRuns);
      
      if (recentRuns.length > 0 && !selectedRun) {
        setSelectedRun(recentRuns[0]);
      }
    } catch (err) {
      console.error('Failed to load runs:', err);
      setError('Failed to load execution history');
    } finally {
      setIsLoading(false);
    }
  };

  const checkSystemHealth = async () => {
    try {
      const health = await agentAPI.checkHealth();
      setHealthStatus(health.status || 'operational');
    } catch (err) {
      setHealthStatus('error');
    }
  };

  const handleCreateRun = async () => {
    if (!prompt.trim()) return;

    setIsCreating(true);
    setError(null);

    // Create optimistic placeholder run immediately
    const placeholderRun = {
      run_id: 'temp-' + Date.now(),
      prompt: prompt.trim(),
      status: 'running',
      plan: null,
      execution_log: [],
      result: null,
      error: null,
      created_at: new Date().toISOString(),
      started_at: new Date().toISOString(),
      completed_at: null,
      isPlaceholder: true
    };

    // Show placeholder immediately
    setRuns([placeholderRun, ...runs]);
    setSelectedRun(placeholderRun);
    const submittedPrompt = prompt;
    setPrompt("");

    try {
      // Create the run via API
      const newRun = await agentAPI.createRun(submittedPrompt);

      // Replace placeholder with actual run
      setRuns(prev => [newRun, ...prev.filter(r => !r.isPlaceholder)]);
      setSelectedRun(newRun);

      // Start polling for updates if run is not immediately completed
      if (newRun.status === 'running' || newRun.status === 'pending') {
        startPolling(newRun.run_id);
      }

    } catch (err) {
      console.error('Failed to create run:', err);
      setError(err.message || 'Failed to create execution');

      // Remove placeholder on error
      setRuns(prev => prev.filter(r => !r.isPlaceholder));
      setSelectedRun(runs[0] || null);
      setPrompt(submittedPrompt); // Restore prompt so user can retry
    } finally {
      setIsCreating(false);
    }
  };

  const startPolling = (runId) => {
    const cleanup = pollRunStatus(
      runId,
      (updatedRun) => {
        // Update run in list
        setRuns(prev => prev.map(r =>
          r.run_id === runId ? updatedRun : r
        ));

        // Update selected run if it's the one being polled
        // Use functional update to get current selectedRun value
        setSelectedRun(current => {
          if (current?.run_id === runId) {
            return updatedRun;
          }
          return current;
        });
      },
      {
        interval: 1000,      // Poll every second
        maxAttempts: 60,     // Stop after 60 seconds
        onError: (err) => {
          console.error('Polling error:', err);
          setError('Lost connection to server');
        }
      }
    );

    // Cleanup will be called automatically when polling stops
    return cleanup;
  };

  const handleSelectRun = async (run) => {
    // If run doesn't have full details, fetch them
    if (!run.plan || !run.execution_log) {
      try {
        const fullRun = await agentAPI.getRun(run.run_id);
        setSelectedRun(fullRun);
        
        // Update in runs list
        setRuns(prev => prev.map(r => 
          r.run_id === run.run_id ? fullRun : r
        ));
      } catch (err) {
        console.error('Failed to load run details:', err);
        setError('Failed to load execution details');
      }
    } else {
      setSelectedRun(run);
    }
  };

  const examplePrompts = [
    "Add a todo to buy milk, then show me all my tasks",
    "Calculate (42 * 8) + 15 and add the result as a todo",
    "Add three tasks: finish report, call dentist, and buy groceries",
    "List all my todos and mark the first one as complete",
    "Calculate 10 + 5, then use the result and subtract it by 15, then add the final number as a todo item like `I have x bugs to fix` (where x is the output of step 2), then show me all my tasks.",
    "Calculate 100 / 10 and add todo Review results then list todos",
    "Add todo Write tests, add todo Review code, then list all todos"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <div className="bg-slate-900/95 border-b border-cyan-900/30 shadow-xl backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-teal-500 rounded-lg flex items-center justify-center shadow-lg">
                <Activity size={24} className="text-slate-900" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-100 uppercase tracking-tight">AI Agent Runtime</h1>
                <p className="text-xs text-cyan-400 font-semibold uppercase tracking-wider">Execution Dashboard</p>
              </div>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-900/30 rounded border border-emerald-600/40">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-lg shadow-emerald-400/50" />
              <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">{healthStatus}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-[1800px] mx-auto px-6 py-8">
        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-orange-900/30 border-2 border-orange-600/40 rounded-lg backdrop-blur-sm">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2 text-orange-400">
                <AlertCircle size={20} />
                <span className="font-semibold">{error}</span>
              </div>
              <button
                onClick={() => setError(null)}
                className="text-orange-400 hover:text-orange-300 transition-colors"
              >
                <XCircle size={18} />
              </button>
            </div>
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <Loader className="animate-spin text-cyan-400 mx-auto mb-4" size={48} />
              <p className="text-slate-400 font-semibold">Loading execution history...</p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-12 gap-6">
          {/* Left Panel - New Execution */}
          <div className="col-span-3 space-y-6">
            {/* Create New Run */}
            <div className="bg-slate-800/50 rounded-lg shadow-xl border border-slate-700/50 p-6 backdrop-blur-sm sticky top-6">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-teal-500 rounded-lg flex items-center justify-center shadow-lg">
                  <PlayCircle size={18} className="text-slate-900" />
                </div>
                <h3 className="text-lg font-bold text-slate-100 uppercase tracking-wide">New Execution</h3>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-bold text-cyan-400 mb-2 uppercase tracking-wider">
                    Task Description
                  </label>
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={(e) => {
                      // CMD+Enter (Mac) or CTRL+Enter (Windows/Linux)
                      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                        e.preventDefault();
                        if (prompt.trim() && !isCreating) {
                          handleCreateRun();
                        }
                      }
                    }}
                    placeholder="Describe what you need the agent to do..."
                    className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent resize-none text-slate-200 placeholder-slate-600 font-medium"
                    rows={4}
                    disabled={isCreating}
                  />
                  <div className="text-xs text-slate-500 mt-2 font-semibold">
                    The agent will analyze and execute your request • Press {navigator.platform.includes('Mac') ? '⌘' : 'Ctrl'}+Enter to execute
                  </div>
                </div>
                
                <button
                  onClick={handleCreateRun}
                  disabled={!prompt.trim() || isCreating}
                  className="w-full bg-gradient-to-r from-cyan-600 to-teal-600 text-slate-900 px-4 py-3 rounded-lg font-bold uppercase tracking-wide hover:from-cyan-500 hover:to-teal-500 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-cyan-500/20 flex items-center justify-center gap-2"
                >
                  {isCreating ? (
                    <>
                      <Loader size={18} className="animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <PlayCircle size={18} />
                      Execute
                    </>
                  )}
                </button>
                
                <div className="pt-4 border-t border-slate-700/50">
                  <div className="text-xs font-bold text-slate-500 mb-2 uppercase tracking-wider flex items-center gap-2">
                    <span>⚡</span> Examples
                  </div>
                  <div className="space-y-1.5">
                    {examplePrompts.map((example, idx) => (
                      <button
                        key={idx}
                        onClick={() => setPrompt(example)}
                        className="w-full text-left text-xs text-slate-400 hover:text-cyan-400 hover:bg-slate-800/50 px-3 py-2 rounded-lg transition-all border border-transparent hover:border-cyan-900/50"
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Center Panel - Execution Details */}
          <div className="col-span-6">
            <div className="bg-slate-800/50 rounded-lg shadow-xl border border-slate-700/50 p-6 min-h-[600px] backdrop-blur-sm">
              <RunDetails run={selectedRun} />
            </div>
          </div>

          {/* Right Panel - Run History */}
          <div className="col-span-3">
            <div className="bg-slate-800/50 rounded-lg shadow-xl border border-slate-700/50 p-6 backdrop-blur-sm sticky top-6">
              <div className="flex items-center gap-2 mb-4">
                <Clock size={18} className="text-slate-400" />
                <h3 className="text-lg font-bold text-slate-100 uppercase tracking-wide">History</h3>
                <span className="ml-auto text-xs bg-slate-700 px-2 py-1 rounded text-slate-400 font-bold">
                  {runs.length}
                </span>
              </div>

              <div className="space-y-2 max-h-[calc(100vh-200px)] overflow-y-auto pr-2">
                {runs.length === 0 ? (
                  <div className="text-center py-8 text-slate-500">
                    <Clock size={32} className="mx-auto mb-3 opacity-50" />
                    <p className="text-sm">No execution history yet</p>
                    <p className="text-xs mt-1">Create your first run to get started</p>
                  </div>
                ) : (
                  runs.map((run) => {
                  const stepCount = run.plan?.steps?.length || 0;
                  const completedSteps = run.execution_log?.filter(e => e.status === 'completed').length || 0;

                  return (
                    <button
                      key={run.run_id}
                      onClick={() => handleSelectRun(run)}
                      className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                        selectedRun?.run_id === run.run_id
                          ? 'border-cyan-600/60 bg-gradient-to-r from-cyan-900/30 to-teal-900/30 shadow-lg shadow-cyan-500/10'
                          : 'border-slate-700/50 bg-slate-800/30 hover:border-slate-600 hover:bg-slate-800/50'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <StatusBadge status={run.status} />
                        <div className="text-xs text-slate-500 font-mono">
                          {new Date(run.created_at).toLocaleTimeString()}
                        </div>
                      </div>

                      <div className="text-sm text-slate-200 line-clamp-2 mb-3 leading-relaxed">
                        {run.prompt}
                      </div>

                      {!run.isPlaceholder && stepCount > 0 && (
                        <div className="flex items-center gap-3 text-xs">
                          <div className="flex items-center gap-1.5 text-cyan-400 font-semibold">
                            <div className="w-5 h-5 rounded bg-cyan-900/40 flex items-center justify-center border border-cyan-700/50">
                              {stepCount}
                            </div>
                            <span className="uppercase tracking-wide">Steps</span>
                          </div>

                        {run.status === 'completed' && (
                          <div className="flex items-center gap-1 text-emerald-400 font-semibold uppercase tracking-wide">
                            <CheckCircle size={12} />
                            <span>Done</span>
                          </div>
                        )}

                        {run.status === 'failed' && (
                          <div className="flex items-center gap-1 text-orange-400 font-semibold uppercase tracking-wide">
                            <XCircle size={12} />
                            <span>Error</span>
                          </div>
                        )}

                        {run.status === 'running' && (
                          <div className="flex items-center gap-1 text-cyan-400 font-semibold">
                            <Loader size={12} className="animate-spin" />
                            <span>{completedSteps}/{stepCount}</span>
                          </div>
                        )}
                        </div>
                      )}

                      <div className="font-mono text-xs text-slate-600 mt-2">
                        {run.run_id}
                      </div>
                    </button>
                  );
                })
                )}
              </div>
            </div>
          </div>
        </div>
        )}
      </div>

      {/* Metrics Dashboard */}
      <MetricsDashboard />

    </div>
  );
}
