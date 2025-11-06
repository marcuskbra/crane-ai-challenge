/**
 * API Service Layer for AI Agent Runtime
 *
 * This module handles all HTTP requests to the backend API.
 * Configure the base URL via environment variable: VITE_API_URL
 */

// Use empty string for proxy (requests go to same origin, then proxied by Vite)
// Or use full URL if VITE_API_URL is set for production
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

/**
 * Generic fetch wrapper with error handling
 */
const fetchAPI = async (endpoint, options = {}) => {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);
    
    // Handle different HTTP status codes
    if (!response.ok) {
      const errorText = await response.text();
      let errorMessage;
      
      try {
        const errorJson = JSON.parse(errorText);
        errorMessage = errorJson.error || errorJson.message || errorText;
      } catch {
        errorMessage = errorText;
      }
      
      throw new Error(`HTTP ${response.status}: ${errorMessage}`);
    }
    
    // Handle 204 No Content
    if (response.status === 204) {
      return null;
    }
    
    return await response.json();
  } catch (error) {
    // Network errors or other issues
    if (error.name === 'TypeError') {
      throw new Error('Network error: Unable to reach the server');
    }
    throw error;
  }
};

/**
 * Agent Runtime API Methods
 */
export const agentAPI = {
  /**
   * Create a new agent execution run
   * @param {string} prompt - The natural language task description
   * @returns {Promise<Object>} The created run object
   */
  createRun: async (prompt) => {
    return fetchAPI('/runs', {
      method: 'POST',
      body: JSON.stringify({ prompt }),
    });
  },

  /**
   * Get the status and details of a specific run
   * @param {string} runId - The unique run identifier
   * @returns {Promise<Object>} The run object with full execution details
   */
  getRun: async (runId) => {
    return fetchAPI(`/runs/${runId}`);
  },

  /**
   * Get a list of recent runs
   * @param {number} limit - Maximum number of runs to return (default: 10)
   * @param {number} offset - Number of runs to skip (for pagination)
   * @returns {Promise<Array>} Array of run objects
   */
  getRecentRuns: async (limit = 10, offset = 0) => {
    const response = await fetchAPI(`/runs?limit=${limit}&offset=${offset}`);
    
    // Handle both array response and object with 'runs' property
    return Array.isArray(response) ? response : (response.runs || []);
  },

  /**
   * Check system health status
   * @returns {Promise<Object>} Health status object
   */
  checkHealth: async () => {
    try {
      return await fetchAPI('/health');
    } catch (error) {
      return { status: 'error', message: error.message };
    }
  },

  /**
   * Get system metrics (performance, usage, statistics)
   * @returns {Promise<Object>} System metrics object
   */
  getMetrics: async () => {
    try {
      return await fetchAPI('/metrics');
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
      return null;
    }
  },

  /**
   * Cancel a running execution (if supported by backend)
   * @param {string} runId - The unique run identifier
   * @returns {Promise<Object>} Updated run object
   */
  cancelRun: async (runId) => {
    return fetchAPI(`/runs/${runId}/cancel`, {
      method: 'POST',
    });
  },
};

/**
 * Polling utility for run status updates
 * @param {string} runId - The run to poll
 * @param {Function} callback - Called with updated run data
 * @param {Object} options - Polling configuration
 * @returns {Function} Cleanup function to stop polling
 */
export const pollRunStatus = (runId, callback, options = {}) => {
  const {
    interval = 1000,        // Poll every 1 second
    maxAttempts = 60,       // Stop after 60 attempts (60 seconds)
    onError = console.error,
  } = options;

  let attempts = 0;
  let timeoutId = null;
  let stopped = false;

  const poll = async () => {
    if (stopped) return;

    try {
      const run = await agentAPI.getRun(runId);
      callback(run);

      // Stop polling if run is completed or failed
      if (run.status !== 'running' && run.status !== 'pending') {
        return;
      }

      // Continue polling if under max attempts
      attempts++;
      if (attempts < maxAttempts) {
        timeoutId = setTimeout(poll, interval);
      }
    } catch (error) {
      onError(error);
      // Continue polling on error (in case of temporary network issues)
      if (attempts < maxAttempts) {
        timeoutId = setTimeout(poll, interval);
      }
    }
  };

  // Start polling
  poll();

  // Return cleanup function
  return () => {
    stopped = true;
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  };
};

/**
 * WebSocket connection for real-time updates (optional feature)
 * @param {string} runId - The run to watch
 * @param {Function} onUpdate - Called when run is updated
 * @param {Function} onError - Called on connection errors
 * @returns {Function} Cleanup function to close connection
 */
export const watchRun = (runId, onUpdate, onError = console.error) => {
  const wsUrl = API_BASE_URL.replace('http', 'ws');
  const ws = new WebSocket(`${wsUrl}/ws/runs/${runId}`);

  ws.onmessage = (event) => {
    try {
      const run = JSON.parse(event.data);
      onUpdate(run);
    } catch (error) {
      onError(error);
    }
  };

  ws.onerror = (error) => {
    onError(new Error('WebSocket connection error'));
  };

  ws.onclose = () => {
    console.log('WebSocket connection closed');
  };

  // Return cleanup function
  return () => {
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close();
    }
  };
};

export default agentAPI;
