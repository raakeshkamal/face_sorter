import React, { useState, useEffect, useCallback, useMemo } from 'react';
import './Training.css';
import ProgressBar from '../components/ProgressBar.jsx';
import ImageCarousel from '../components/ImageCarousel.jsx';
import FolderPicker from '../components/FolderPicker.jsx';
import { apiService } from '../services/api';
import websocketService from '../services/websocket';

function Training() {
  const [form, setForm] = useState({
    source_dir: '',
    noface_dir: '',
    broken_dir: '',
    duplicates_dir: '',
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

  // Image carousel state
  const [images, setImages] = useState([]);
  const [carouselIndex, setCarouselIndex] = useState(0);
  const [isCarouselPaused, setIsCarouselPaused] = useState(false);

  const [showFolderPicker, setShowFolderPicker] = useState(false);
  const [currentField, setCurrentField] = useState('');
  const [status, setStatus] = useState('idle'); // 'idle', 'active', 'complete', 'cancelled', 'failed'

  // Check for active training session on component mount
  useEffect(() => {
    const checkActiveSession = async () => {
      try {
        const activeSessions = await apiService.getActiveSessions();
        const trainingSession = activeSessions.find(s => s.operation_type === 'training');
        if (trainingSession) {
          // Only reconnect WebSocket for sessions with status 'running'
          if (trainingSession.status === 'running') {
            setTaskId(trainingSession.task_id);
            setOperationStarted(true);

            // Restore progress from session
            if (trainingSession.progress) {
              setCurrent(trainingSession.progress.current || 0);
              setTotal(trainingSession.progress.total || 0);
              setCurrentStatus(trainingSession.progress.status || '');
              setCurrentItem(trainingSession.progress.current_item || '');
            }

            setStatus('active');

            // Reconnect to WebSocket only for running sessions
            websocketService.connect('training', trainingSession.task_id, handleMessage, handleError);
          } else {
            // Session is cancelled, completed, or failed - don't reconnect
            console.log(`Session ${trainingSession.task_id} has status ${trainingSession.status}, not reconnecting`);
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
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleMessage = (data) => {
    switch (data.type) {
      case 'progress':
        setCurrent(data.progress.current);
        setTotal(data.progress.total);
        setCurrentStatus(data.progress.status);
        setCurrentItem(data.progress.current_item || '');

        // Handle image data for carousel (only display images with faces)
        if (data.progress.image_data && data.progress.image_data.det_score !== undefined && !isCarouselPaused) {
          setImages((prevImages) => {
            const newImage = data.progress.image_data;
            // Check if this image is already in our list
            const imageExists = prevImages.some((img) => img.filename === newImage.filename);

            if (!imageExists) {
              // Add new image to the carousel and keep only last 50
              const updatedImages = [...prevImages, newImage].slice(-50);
              // Update carousel index to always point to the newest image
              setCarouselIndex(updatedImages.length - 1);
              return updatedImages;
            }
            return prevImages;
          });
        }

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
        window.alert(`Training failed: ${data.error.message || 'Unknown error'}`);
        break;
      default:
        break;
    }
  };

  const handleError = (wsError) => {
    console.error('[Training] WebSocket error:', wsError);
    window.alert('WebSocket connection error. Progress updates may not be available.');
  };

  const startTraining = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setStatus('active');
      const response = await apiService.startTraining(form);
      setTaskId(response.task_id);
      setOperationStarted(true);

      websocketService.connect('training', response.task_id, handleMessage, handleError);
    } catch (error) {
      console.error('Failed to start training:', error);
      window.alert('Failed to start training. Please check your configuration and try again.');
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
      setImages([]);
      setCarouselIndex(0);
      setIsCarouselPaused(false);

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
    setImages([]);
    setCarouselIndex(0);
    setIsCarouselPaused(false);
  };

  return (
    <div className="training-view">
      <div className="view-header">
        <h1>🎯 Training</h1>
        <p className="subtitle">Detect faces and generate embeddings</p>
      </div>

      {!operationStarted ? (
        <div className="training-form card">
          <h2 className="form-title">Training Configuration</h2>
          <form onSubmit={startTraining}>
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
                    placeholder="/path/to/images"
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
                <p className="form-help">Directory containing images to process</p>
              </div>

              <div className="form-group">
                <label className="form-label">No-Face Directory</label>
                <div className="input-with-browse">
                  <input
                    type="text"
                    name="noface_dir"
                    value={form.noface_dir}
                    onChange={handleChange}
                    className="form-input"
                    placeholder="/path/to/noface"
                  />
                  <button
                    type="button"
                    className="browse-btn"
                    onClick={() => openFolderPicker('noface_dir')}
                  >
                    📁 Browse
                  </button>
                </div>
                <p className="form-help">Directory for images without faces</p>
              </div>

              <div className="form-group">
                <label className="form-label">Broken Images Directory</label>
                <div className="input-with-browse">
                  <input
                    type="text"
                    name="broken_dir"
                    value={form.broken_dir}
                    onChange={handleChange}
                    className="form-input"
                    placeholder="/path/to/broken"
                  />
                  <button
                    type="button"
                    className="browse-btn"
                    onClick={() => openFolderPicker('broken_dir')}
                  >
                    📁 Browse
                  </button>
                </div>
                <p className="form-help">Directory for corrupted images</p>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Duplicates Directory</label>
              <div className="input-with-browse">
                <input
                  type="text"
                  name="duplicates_dir"
                  value={form.duplicates_dir}
                  onChange={handleChange}
                  className="form-input"
                  placeholder="/path/to/duplicates"
                />
                <button
                  type="button"
                  className="browse-btn"
                  onClick={() => openFolderPicker('duplicates_dir')}
                >
                  📁 Browse
                </button>
              </div>
              <p className="form-help">Directory for duplicate images (will be skipped)</p>
            </div>

            <div className="form-actions">
              <button type="submit" className="btn btn-primary btn-large" disabled={loading}>
                <span className="btn-icon">🎯</span>
                <span>{loading ? 'Starting...' : 'Start Training'}</span>
              </button>
            </div>
          </form>
        </div>
      ) : (
        <div className="progress-section">
          {/* Image Carousel for visual feedback */}
          {status === 'active' && images.length > 0 && (
            <ImageCarousel
              images={images}
              currentIndex={carouselIndex}
              totalImages={total}
              autoAdvance={!isCarouselPaused}
              onPauseToggle={setIsCarouselPaused}
              className="carousel-container"
            />
          )}

          {/* Progress bar for detailed information */}
          <ProgressBar
            operationType="Training"
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
        <h3 className="info-title">About Training</h3>
        <div className="info-content">
          <p className="info-paragraph">
            <strong>Training</strong> performs face detection on your images using
            InsightFace and generates 512-dimensional face embeddings that can be used
            for recognition and clustering.
          </p>
          <ul className="info-list">
            <li>✅ Detects faces in images</li>
            <li>✅ Generates 512-dim embeddings</li>
            <li>✅ Extracts face metadata (age, gender, landmarks)</li>
            <li>✅ Moves images without faces to noface directory</li>
            <li>✅ Real-time progress tracking</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Training;