import React, { useState, useEffect } from 'react';
import './Cleaning.css';
import ProgressBar from '../components/ProgressBar.jsx';
import FolderPicker from '../components/FolderPicker.jsx';
import { apiService } from '../services/api';
import websocketService from '../services/websocket';

function Cleaning() {
  const [cleanForm, setCleanForm] = useState({
    source_dir: '',
    batch_size: 25,
    img_prefix: 'IMG',
    quality: 95,
    recursive: true,
    start_index: 0,
  });

  const [dedupForm, setDedupForm] = useState({
    source_dir: '',
    dedup_threshold: 0.98,
    dedup_batch_size: 32,
    dedup_model_name: 'openai/clip-vit-base-patch32',
    dedup_force_recompute: false,
  });

  // 'idle', 'cleaning', or 'deduplicating'
  const [activeOperationType, setActiveOperationType] = useState('idle');
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
  const [currentField, setCurrentField] = useState({ form: '', field: '' });

  // Check for active session on component mount
  useEffect(() => {
    const checkActiveSession = async () => {
      try {
        const activeSessions = await apiService.getActiveSessions();
        const activeSession = activeSessions.find(s => s.operation_type === 'cleaning' || s.operation_type === 'deduplicating');
        
        if (activeSession) {
          if (activeSession.status === 'running') {
            setTaskId(activeSession.task_id);
            setActiveOperationType(activeSession.operation_type);

            // Restore progress from session
            if (activeSession.progress) {
              setCurrent(activeSession.progress.current || 0);
              setTotal(activeSession.progress.total || 0);
              setCurrentStatus(activeSession.progress.status || '');
              setCurrentItem(activeSession.progress.current_item || '');
            }

            setStatus('active');

            // Reconnect to WebSocket
            websocketService.connect(activeSession.operation_type, activeSession.task_id, handleMessage, handleError);
          } else {
            console.log(`Session ${activeSession.task_id} has status ${activeSession.status}, not reconnecting`);
          }
        }
      } catch (error) {
        console.error('Failed to check active sessions:', error);
      }
    };

    checkActiveSession();

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
    } else if (activeOperationType !== 'idle' && currentStatus !== '') {
      setStatus('active');
    }
  }, [currentStatus, activeOperationType]);

  const openFolderPicker = (formName, fieldName) => {
    setCurrentField({ form: formName, field: fieldName });
    setShowFolderPicker(true);
  };

  const handleFolderSelect = (path) => {
    if (currentField.form === 'clean') {
      setCleanForm((prev) => ({ ...prev, [currentField.field]: path }));
    } else if (currentField.form === 'dedup') {
      setDedupForm((prev) => ({ ...prev, [currentField.field]: path }));
    }
    setShowFolderPicker(false);
    setCurrentField({ form: '', field: '' });
  };

  const handleFolderCancel = () => {
    setShowFolderPicker(false);
    setCurrentField({ form: '', field: '' });
  };

  const handleCleanChange = (e) => {
    const { name, value, type, checked } = e.target;
    setCleanForm((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleDedupChange = (e) => {
    const { name, value, type, checked } = e.target;
    setDedupForm((prev) => ({
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
        window.alert(`Operation failed: ${data.error.message || 'Unknown error'}`);
        break;
      default:
        break;
    }
  };

  const handleError = (wsError) => {
    console.error('[Operation] WebSocket error:', wsError);
    window.alert('WebSocket connection error. Progress updates may not be available.');
  };

  const startCleaning = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setStatus('active');
      const response = await apiService.startCleaning(cleanForm);
      setTaskId(response.task_id);
      setActiveOperationType('cleaning');

      websocketService.connect('cleaning', response.task_id, handleMessage, handleError);
    } catch (error) {
      console.error('Failed to start cleaning:', error);
      window.alert('Failed to start cleaning. Please check your configuration and try again.');
      setStatus('idle');
      setActiveOperationType('idle');
    } finally {
      setLoading(false);
    }
  };

  const startDeduplication = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setStatus('active');
      const response = await apiService.startDeduplication(dedupForm);
      setTaskId(response.task_id);
      setActiveOperationType('deduplicating');

      websocketService.connect('deduplicating', response.task_id, handleMessage, handleError);
    } catch (error) {
      console.error('Failed to start deduplication:', error);
      window.alert('Failed to start deduplication. Please check your configuration and try again.');
      setStatus('idle');
      setActiveOperationType('idle');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async () => {
    if (!taskId) return;

    try {
      setCancelling(true);
      await apiService.cancelSession(taskId);
      websocketService.disconnect();

      setStatus('cancelled');
      setCurrentStatus('Cancelled');
      setActiveOperationType('idle');
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
    setActiveOperationType('idle');
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
        <h1>🧹 Dataset Operations</h1>
        <p className="subtitle">Clean, standardize, and deduplicate your images</p>
      </div>

      {activeOperationType === 'idle' ? (
        <div className="operations-grid">
          {/* Cleaning Card */}
          <div className="cleaning-form card">
            <h2 className="form-title">Cleaning</h2>
            <p className="form-help" style={{ marginBottom: "1rem" }}>Validates images, converts to RGB JPEG, and applies sequential naming.</p>
            <form onSubmit={startCleaning} style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
              <div className="form-group">
                <label className="form-label">Source Directory</label>
                <div className="input-with-browse">
                  <input
                    type="text"
                    name="source_dir"
                    value={cleanForm.source_dir}
                    onChange={handleCleanChange}
                    className="form-input"
                    placeholder="/path/to/raw/images"
                    required
                  />
                  <button
                    type="button"
                    className="browse-btn"
                    onClick={() => openFolderPicker('clean', 'source_dir')}
                  >
                    📁 Browse
                  </button>
                </div>
              </div>

              <div className="form-grid grid-2">
                <div className="form-group">
                  <label className="form-label">Batch Size</label>
                  <input
                    type="number"
                    name="batch_size"
                    value={cleanForm.batch_size}
                    onChange={handleCleanChange}
                    className="form-input"
                    min="1"
                    max="100"
                    placeholder="25"
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Image Prefix</label>
                  <input
                    type="text"
                    name="img_prefix"
                    value={cleanForm.img_prefix}
                    onChange={handleCleanChange}
                    className="form-input"
                    placeholder="IMG"
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">JPEG Quality</label>
                  <input
                    type="number"
                    name="quality"
                    value={cleanForm.quality}
                    onChange={handleCleanChange}
                    className="form-input"
                    min="1"
                    max="100"
                    placeholder="95"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Starting Index</label>
                  <input
                    type="number"
                    name="start_index"
                    value={cleanForm.start_index}
                    onChange={handleCleanChange}
                    className="form-input"
                    min="0"
                    placeholder="1"
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label flex-row">
                  <input
                    type="checkbox"
                    name="recursive"
                    checked={cleanForm.recursive}
                    onChange={handleCleanChange}
                    className="form-checkbox"
                  />
                  <span>Scan directories recursively</span>
                </label>
              </div>

              <div className="form-actions" style={{ marginTop: "auto", paddingTop: "1rem" }}>
                <button type="submit" className="btn btn-primary btn-large" style={{ width: "100%" }} disabled={loading}>
                  <span className="btn-icon">🧹</span>
                  <span>{loading ? 'Starting...' : 'Start Cleaning'}</span>
                </button>
              </div>
            </form>
          </div>

          {/* Deduplication Card */}
          <div className="cleaning-form card">
            <h2 className="form-title">Deduplication</h2>
            <p className="form-help" style={{ marginBottom: "1rem" }}>Finds and isolates identical or near-identical images.</p>
            <form onSubmit={startDeduplication} style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
              <div className="form-group">
                <label className="form-label">Source Directory</label>
                <div className="input-with-browse">
                  <input
                    type="text"
                    name="source_dir"
                    value={dedupForm.source_dir}
                    onChange={handleDedupChange}
                    className="form-input"
                    placeholder="/path/to/cleaned/images"
                    required
                  />
                  <button
                    type="button"
                    className="browse-btn"
                    onClick={() => openFolderPicker('dedup', 'source_dir')}
                  >
                    📁 Browse
                  </button>
                </div>
              </div>

              <div className="form-grid grid-2">
                <div className="form-group">
                  <label className="form-label">Similarity Threshold</label>
                  <input
                    type="number"
                    name="dedup_threshold"
                    value={dedupForm.dedup_threshold}
                    onChange={handleDedupChange}
                    className="form-input"
                    step="0.01"
                    min="0"
                    max="1"
                    placeholder="0.98"
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Batch Size</label>
                  <input
                    type="number"
                    name="dedup_batch_size"
                    value={dedupForm.dedup_batch_size}
                    onChange={handleDedupChange}
                    className="form-input"
                    min="1"
                    max="100"
                    placeholder="32"
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Model Name</label>
                <input
                  type="text"
                  name="dedup_model_name"
                  value={dedupForm.dedup_model_name}
                  onChange={handleDedupChange}
                  className="form-input"
                  placeholder="openai/clip-vit-base-patch32"
                />
              </div>

              <div className="form-group">
                <label className="form-label flex-row">
                  <input
                    type="checkbox"
                    name="dedup_force_recompute"
                    checked={dedupForm.dedup_force_recompute}
                    onChange={handleDedupChange}
                    className="form-checkbox"
                  />
                  <span>Force recompute embeddings</span>
                </label>
              </div>

              <div className="form-actions" style={{ marginTop: "auto", paddingTop: "1rem" }}>
                <button type="submit" className="btn btn-secondary btn-large" style={{ width: "100%" }} disabled={loading}>
                  <span className="btn-icon">👯</span>
                  <span>{loading ? 'Starting...' : 'Start Deduplication'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      ) : (
        <div className="progress-section">
          <ProgressBar
            operationType={activeOperationType === 'cleaning' ? 'Cleaning' : 'Deduplication'}
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
        fieldType="directory"
        initialPath={
          currentField.form === 'clean' ? cleanForm[currentField.field] :
          currentField.form === 'dedup' ? dedupForm[currentField.field] : ''
        }
        onSelect={handleFolderSelect}
        onCancel={handleFolderCancel}
      />
    </div>
  );
}

export default Cleaning;
