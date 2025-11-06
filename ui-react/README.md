# Agent Runtime Visualization Dashboard

A React-based UI dashboard for visualizing the crane-challenge agent runtime execution flow in real-time.

> **⚠️ Important Note**: This is a **visualization tool** created for development and demonstration purposes. It is **NOT intended as a production-ready frontend** and was not part of the core assignment requirements. Use it to understand how the agent orchestrator executes plans step-by-step.

## Features

- **Real-Time Execution Monitoring**: Watch agent execution unfold step-by-step with WebSocket-like polling
- **Optimistic UI**: Instant feedback when executing prompts before API responds
- **Step Timeline Visualization**: See the execution flow with timing, status, and tool usage
- **Custom Output Renderers**: Specialized displays for TodoStore and Calculator outputs
- **System Metrics Dashboard**: Track token usage, execution times, and retry attempts
- **Execution History**: Browse and replay previous execution runs

## Tech Stack

- **React 18**: UI framework with hooks
- **Vite**: Fast development server and build tool
- **Tailwind CSS**: Utility-first styling
- **Lucide React**: Icon library for clean UI elements

## Setup

### Prerequisites

- Node.js 18+ and npm
- Backend API running at `http://localhost:8000` (see main project README)

### Installation

```bash
# Install dependencies
npm install
```

### Development

```bash
# Start development server
npm run dev

# Frontend available at http://localhost:3000
```

### Build

```bash
# Build for production (visualization only)
npm run build

# Output in dist/
```

## Project Structure

```
ui-react/
├── src/
│   ├── components/
│   │   └── AgentRuntimeUI.jsx    # Main dashboard component
│   ├── App.jsx                   # App root
│   ├── index.css                 # Tailwind styles
│   └── main.jsx                  # Entry point
├── public/                       # Static assets
├── index.html                    # HTML template
├── package.json                  # Dependencies
├── vite.config.js               # Vite configuration
└── tailwind.config.js           # Tailwind configuration
```

## Usage

1. **Start the Backend**: Ensure the Python API is running at `http://localhost:8000`
2. **Start the Frontend**: Run `npm run dev` to start the visualization dashboard
3. **Execute Prompts**: Enter a natural language prompt and click "Execute"
4. **Watch Execution**: See the agent plan and execute steps in real-time
5. **Browse History**: Click on previous runs to see their execution details

## Component Overview

### AgentRuntimeUI (Main Component)

The main dashboard component with several key sections:

- **Execution Panel**: Prompt input, execute button, and current run display
- **Plan Visualization**: Shows the generated plan with step details
- **Step Timeline**: Real-time execution flow with status indicators
- **Output Display**: Custom renderers for different tool outputs
- **Technical Metadata**: System metrics and performance data
- **History Panel**: List of all execution runs with filtering

### Key Features Implementation

#### Optimistic UI Pattern
```javascript
// Creates placeholder run immediately for instant feedback
const optimisticRun = {
  run_id: `placeholder-${Date.now()}`,
  prompt: promptText,
  status: "pending",
  plan: null,
  isPlaceholder: true
};
```

#### Real-Time Polling
```javascript
// Simulates WebSocket with interval polling
useEffect(() => {
  if (selectedRunId && !selectedRun?.isPlaceholder) {
    const interval = setInterval(() => {
      fetchRunDetails(selectedRunId);
    }, 1000);
    return () => clearInterval(interval);
  }
}, [selectedRunId]);
```

#### Custom Output Renderers
- **TodoStore**: Renders todos with status badges and checkboxes
- **Calculator**: Displays calculations with formatted results
- **Default**: JSON pretty-print for other tool outputs

## Styling Conventions

- **Dark Theme**: Cyberpunk-inspired with cyan/purple accents
- **Status Colors**:
  - 🟢 Green: Completed successfully
  - 🔵 Blue: Running/In progress
  - 🟡 Yellow: Pending/Waiting
  - 🔴 Red: Failed/Error
- **Glassmorphism**: Subtle transparency and backdrop blur effects
- **Responsive Grid**: Adapts to different screen sizes

## API Integration

The UI expects the following API endpoints:

- `POST /api/v1/runs`: Create new execution run
- `GET /api/v1/runs`: List all execution runs
- `GET /api/v1/runs/{run_id}`: Get execution run details

See the main project API documentation for details.

## Development Notes

### Performance Considerations

- Polling interval set to 1 second (can be adjusted)
- History limited to recent runs (currently no pagination)
- Component re-renders optimized with React.memo patterns

### Known Limitations

- No WebSocket support (uses polling instead)
- No authentication/authorization
- No error boundary implementation
- Limited accessibility features
- No unit tests (visualization tool only)

## Future Enhancements (If Needed)

- [ ] WebSocket integration for true real-time updates
- [ ] Pagination for execution history
- [ ] Export execution runs to JSON
- [ ] Dark/light theme toggle
- [ ] Keyboard shortcuts for common actions
- [ ] Accessibility improvements (ARIA labels, keyboard navigation)
- [ ] Error boundary and better error handling
- [ ] Unit tests with React Testing Library

## Troubleshooting

### Frontend won't start
```bash
# Clean install
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Backend connection issues
- Verify backend is running at `http://localhost:8000`
- Check browser console for CORS errors
- Ensure API endpoints are accessible

### Styling issues
```bash
# Rebuild Tailwind
npm run build
```

## License

Part of the crane-challenge project - see main project LICENSE for details.
