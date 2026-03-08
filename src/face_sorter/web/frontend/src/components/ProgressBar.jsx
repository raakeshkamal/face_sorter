import React, { useMemo } from 'react';
import './ProgressBar.css';

function ProgressBar({
  operationType = 'Operation',
  taskId,
  cancellable = false,
  idleText = 'Ready to start',
  current = 0,
  total = 0,
  currentStatus = '',
  currentItem = '',
  logs = [],
  onCancel,
  onReset,
  status = 'idle', // 'idle', 'active', 'complete', 'cancelled', 'failed'
  cancelling = false
}) {
  const started = status !== 'idle';
  const completed = status === 'complete';
  const active = status === 'active';
  const error = status === 'failed';
  const cancelled = status === 'cancelled';

  const loading = active && !completed && !error && !cancelled;

  const progressPercentage = useMemo(() => {
    if (total === 0) return 0;
    return Math.round((current / total) * 100);
  }, [current, total]);

  const eta = useMemo(() => {
    if (!loading || current === 0) return '';
    const remaining = total - current;
    if (remaining <= 0) return 'Complete';
    return `${remaining} remaining`;
  }, [loading, current, total]);

  if (!started) {
    return (
      <div className="progress-bar-container">
        <div className="progress-idle">
          <span className="idle-icon">⏳</span>
          <span className="idle-text">{idleText}</span>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="progress-bar-container">
        <div className="progress-active">
          <div className="progress-header">
            <div className="progress-info">
              <span className="progress-title">{operationType}</span>
              <span className="progress-status">{currentStatus}</span>
            </div>
            {onCancel && (
              <button
                className="cancel-btn"
                onClick={onCancel}
                disabled={cancelling}
                title={cancelling ? "Cancelling..." : "Cancel operation"}
              >
                {cancelling ? "⏳" : "✕"}
              </button>
            )}
          </div>

          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progressPercentage}%` }}>
              <div className="progress-glow"></div>
            </div>
          </div>

          <div className="progress-details">
            <div className="progress-stat">
              <span className="stat-label">Progress:</span>
              <span className="stat-value">{current} / {total}</span>
            </div>
            <div className="progress-stat">
              <span className="stat-label">Percentage:</span>
              <span className="stat-value">{progressPercentage}%</span>
            </div>
            {currentItem && (
              <div className="progress-stat">
                <span className="stat-label">Current:</span>
                <span className="stat-value">{currentItem}</span>
              </div>
            )}
            {eta && (
              <div className="progress-stat">
                <span className="stat-label">ETA:</span>
                <span className="stat-value">{eta}</span>
              </div>
            )}
          </div>

          {logs.length > 0 && (
            <div className="progress-logs">
              {logs.map((log, index) => (
                <div key={index} className="log-entry">
                  <span className="log-time">{log.time}</span>
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  if (completed) {
    return (
      <div className="progress-bar-container">
        <div className="progress-complete">
          <div className="complete-icon">✓</div>
          <h3 className="complete-title">{operationType} Complete</h3>
          <div className="complete-summary">
            <div className="summary-item">
              <span className="summary-label">Processed:</span>
              <span className="summary-value">{current} / {total}</span>
            </div>
          </div>
          <button className="btn btn-primary" onClick={onReset}>
            Start New Operation
          </button>
        </div>
      </div>
    );
  }

  if (cancelled) {
    return (
      <div className="progress-bar-container">
        <div className="progress-error">
          <div className="error-icon">🛑</div>
          <h3 className="error-title">{operationType} Cancelled</h3>
          <p className="error-message">The operation was cancelled by the user.</p>
          <div className="complete-summary">
            <div className="summary-item">
              <span className="summary-label">Processed:</span>
              <span className="summary-value">{current} / {total}</span>
            </div>
          </div>
          <button className="btn btn-primary" onClick={onReset}>
            Start New Operation
          </button>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="progress-bar-container">
        <div className="progress-error">
          <div className="error-icon">⚠</div>
          <h3 className="error-title">{operationType} Failed</h3>
          <p className="error-message">An error occurred during the operation.</p>
          <button className="btn btn-primary" onClick={onReset}>
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return null;
}

export default ProgressBar;