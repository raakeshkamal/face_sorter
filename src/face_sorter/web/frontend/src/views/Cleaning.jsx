import React, { useState, useEffect } from 'react';
import './Cleaning.css';
import ProgressBar from '../components/ProgressBar.jsx';
import FolderPicker from '../components/FolderPicker.jsx';
import { apiService } from '../services/api';
import websocketService from '../services/websocket';

function Cleaning() {
  const [form, setForm] = useState({
    source_dir: '',
    batch_size: 25,
    img_prefix: 'IMG',
    quality: 95,
    recursive: true,
    start_index: 0,
  });

  const [operationStarted, setOperationStarted] = useState(false);
  const [taskId, setTaskId] = useState('');
  const [loading, setLoading] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  // Progress state
  const [current, setCurrent] = useState(0);
  const [total, setTotal] = useState(0);
  const [currentStatus, setCurrentStatus] = useState('');
  const [currentItem, setCurrentItem] = useState('');
  const [logs, setLogs] = useState([]);

  const [status, setStatus] = useState('idle'); // 'idle', 'active', 'complete', 'cancelled', 'failed'

  const [showFolderPicker, setShowFolderPicker] = useState(false);
  const [currentField, setCurrentField] = useState('');

  // Check for active cleaning session on component mount
  useEffect(() => {
    const checkActiveSession = async () => {
      try {
        const activeSessions = await apiService.getActiveSessions();
        const cleaningSession = activeSessions.find(s => s.operation_type === 'cleaning');
        if (cleaningSession) {
          // Only reconnect WebSocket for sessions with status 'running'
          if (cleaningSession.status === 'running') {
            setTaskId(cleaningSession.task_id);
            setOperationStarted(true);

            // Restore progress from session
            if (cleaningSession.progress) {
              setCurrent(cleaningSession.progress.current || 0);
              setTotal(cleaningSession.progress.total || 0);
              setCurrentStatus(cleaningSession.progress.status || '');
              setCurrentItem(cleaningSession.progress.current_item || '');
            }

            setStatus('active');

            // Reconnect to WebSocket only for running sessions
            websocketService.connect('cleaning', cleaningSession.task_id, handleMessage, handleError);
          } else {
            // Session is cancelled, completed, or failed - don't reconnect
            console.log(`Session ${cleaningSession.task_id} has status ${cleaningSession.status}, not reconnecting`);
          }
        }
      } catch (error) {
        console.error('Failed to check active sessions:', error);
      }
    };

    checkActiveSession();

    // Cleanup WebSocket on unmount
    return () => {
      websocketService.disconnect();
    };
  }, []);

  // Update status based on currentStatus
  useEffect(() => {
    if (currentStatus === 'Complete') {
      setStatus('complete');
    } else if (currentStatus === 'Cancelled') {
      setStatus('cancelled');
    } else if (currentStatus === 'Failed') {
      setStatus('failed');
    } else if (operationStarted && currentStatus !== '') {
      setStatus('active');
    }
  }, [currentStatus, operationStarted]);

  const openFolderPicker = (field) => {
    setCurrentField(field);
    setShowFolderPicker(true);
  };

  const handleFolderSelect = (path) => {
    if (currentField) {
      setForm((prev) => ({ ...prev, [currentField]: path }));
    }
    setShowFolderPicker(false);
    setCurrentField('');
  };

  const handleFolderCancel = () => {
    setShowFolderPicker(false);
    setCurrentField('');
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleMessage = (data) => {
    switch (data.type) {
      case 'progress':
        setCurrent(data.progress.current);
        setTotal(data.progress.total);
        setCurrentStatus(data.progress.status);
        setCurrentItem(data.progress.current_item || '');

        setLogs((prevLogs) => {
          const newLog = {
            time: new Date().toLocaleTimeString(),
            message: `${data.progress.status}: ${data.progress.current_item || ''}`,
          };
          const updatedLogs = [newLog, ...prevLogs];
          return updatedLogs.slice(0, 20);
        });
        break;

      case 'complete':
        setCurrentStatus('Complete');
        setStatus('complete');
        break;

      case 'error':
        setCurrentStatus('Failed');
        setStatus('failed');
        window.alert(`Cleaning failed: ${data.error.message || 'Unknown error'}`);
        break;
      default:
        break;
    }
  };

  const handleError = (wsError) => {
    console.error('[Cleaning] WebSocket error:', wsError);
    window.alert('WebSocket connection error. Progress updates may not be available.');
  };

  const startCleaning = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setStatus('active');
      const response = await apiService.startCleaning(form);
      setTaskId(response.task_id);
      setOperationStarted(true);

      websocketService.connect('cleaning', response.task_id, handleMessage, handleError);
    } catch (error) {
      console.error('Failed to start cleaning:', error);
      window.alert('Failed to start cleaning. Please check your configuration and try again.');
      setStatus('idle');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async () => {
    if (!taskId) return;

    try {
      // Show loading state
      setCancelling(true);

      // Call cancel API
      await apiService.cancelSession(taskId);

      // Disconnect WebSocket after successful cancellation
      websocketService.disconnect();

      // Update state
      setStatus('cancelled');
      setCurrentStatus('Cancelled');
      setOperationStarted(false);
      setTaskId('');
      setCurrent(0);
      setTotal(0);
      setCurrentItem('');
      setLogs([]);

    } catch (error) {
      console.error('Failed to cancel operation:', error);
      window.alert('Failed to cancel operation. Please try again.');
    } finally {
      setCancelling(false);
    }
  };

  const handleReset = () => {
    websocketService.disconnect();
    setOperationStarted(false);
    setTaskId('');
    setCurrent(0);
    setTotal(0);
    setCurrentStatus('');
    setCurrentItem('');
    setLogs([]);
  };

  return (
    <div className="cleaning-view">
      <div className="view-header">
        <h1>🧹 Cleaning</h1>
        <p className="subtitle">Validate and standardize image dataset</p>
      </div>

      {!operationStarted ? (
        <div className="cleaning-form card">
          <h2 className="form-title">Cleaning Configuration</h2>
          <form onSubmit={startCleaning}>
            <div className="form-grid grid-2">
              <div className="form-group">
                <label className="form-label">Source Directory</label>
                <div className="input-with-browse">
                  <input
                    type="text"
                    name="source_dir"
                    value={form.source_dir}
                    onChange={handleChange}
                    className="form-input"
                    placeholder="/path/to/raw/images"
                    required
                  />
                  <button
                    type="button"
                    className="browse-btn"
                    onClick={() => openFolderPicker('source_dir')}
                  >
                    📁 Browse
                  </button>
                </div>
                <p className="form-help">Directory containing images to clean. Output and broken directories will be created in the same parent folder.</p>
              </div>

              <div className="form-group">
                <label className="form-label">Batch Size</label>
                <input
                  type="number"
                  name="batch_size"
                  value={form.batch_size}
                  onChange={handleChange}
                  className="form-input"
                  min="1"
                  max="100"
                  placeholder="25"
                />
                <p className="form-help">Images to process at once (1-100)</p>
              </div>

              <div className="form-group">
                <label className="form-label">Image Prefix</label>
                <input
                  type="text"
                  name="img_prefix"
                  value={form.img_prefix}
                  onChange={handleChange}
                  className="form-input"
                  placeholder="IMG"
                />
                <p className="form-help">Prefix for output filenames (e.g., IMG_001.jpg)</p>
              </div>

              <div className="form-group">
                <label className="form-label">JPEG Quality</label>
                <input
                  type="number"
                  name="quality"
                  value={form.quality}
                  onChange={handleChange}
                  className="form-input"
                  min="1"
                  max="100"
                  placeholder="95"
                />
                <p className="form-help">JPEG quality (1-100, higher = better quality)</p>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label flex-row">
                <input
                  type="checkbox"
                  name="recursive"
                  checked={form.recursive}
                  onChange={handleChange}
                  className="form-checkbox"
                />
                <span>Scan directories recursively</span>
              </label>
              <p className="form-help">Search subdirectories for images</p>
            </div>

            <div className="form-group">
              <label className="form-label">Starting Index</label>
              <input
                type="number"
                name="start_index"
                value={form.start_index}
                onChange={handleChange}
                className="form-input"
                min="0"
                placeholder="1"
              />
              <p className="form-help">Starting number for sequential naming (0 for auto-detect)</p>
            </div>

            <div className="form-actions">
              <button type="submit" className="btn btn-primary btn-large" disabled={loading}>
                <span className="btn-icon">🧹</span>
                <span>{loading ? 'Starting...' : 'Start Cleaning'}</span>
              </button>
            </div>
          </form>
        </div>
      ) : (
        <div className="progress-section">
          <ProgressBar
            operationType="Cleaning"
            taskId={taskId}
            current={current}
            total={total}
            currentStatus={currentStatus}
            currentItem={currentItem}
            logs={logs}
            idleText="Initializing..."
            onCancel={handleCancel}
            onReset={handleReset}
            status={status}
            cancelling={cancelling}
          />
        </div>
      )}

      <FolderPicker
        show={showFolderPicker}
        fieldType={currentField}
        initialPath={form[currentField] || ''}
        onSelect={handleFolderSelect}
        onCancel={handleFolderCancel}
      />

      <div className="info-section card">
        <h3 className="info-title">About Cleaning</h3>
        <div className="info-content">
          <p className="info-paragraph">
            <strong>Cleaning</strong> validates your image dataset, converts all images to RGB JPEG format,
            and saves them with sequential naming to a flat output directory.
          </p>
          <ul className="info-list">
            <li>✅ Validates image integrity</li>
            <li>✅ Converts to RGB JPEG format</li>
            <li>✅ Applies sequential naming (IMG_001.jpg, IMG_002.jpg, etc.)</li>
            <li>✅ Moves broken images to separate directory</li>
            <li>✅ Automatically creates output and broken folders in same parent</li>
            <li>✅ Real-time progress tracking</li>
            <li>✅ Supports various image formats (JPG, PNG, BMP, etc.)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Cleaning;