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
  });

  const [operationStarted, setOperationStarted] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
  const [stabilityComplete, setStabilityComplete] = useState(false);
  const [taskId, setTaskId] = useState('');
  const [loading, setLoading] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [operationType, setOperationType] = useState(''); // 'training_stability' or 'training_faces'

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
        const stabilitySession = activeSessions.find(s => s.operation_type === 'training_stability');
        const facesSession = activeSessions.find(s => s.operation_type === 'training_faces');

        if (stabilitySession) {
          setOperationType('training_stability');
          // Only reconnect WebSocket for sessions with status 'running'
          if (stabilitySession.status === 'running') {
            setTaskId(stabilitySession.task_id);
            setOperationStarted(true);
            setCurrentStep(1);

            // Restore progress from session
            if (stabilitySession.progress) {
              setCurrent(stabilitySession.progress.current || 0);
              setTotal(stabilitySession.progress.total || 0);
              setCurrentStatus(stabilitySession.progress.status || '');
              setCurrentItem(stabilitySession.progress.current_item || '');
            }

            setStatus('active');

            // Reconnect to WebSocket only for running sessions
            websocketService.connect('training_stability', stabilitySession.task_id, handleMessage, handleError);
          } else {
            // Session is cancelled, completed, or failed - don't reconnect
            console.log(`Session ${stabilitySession.task_id} has status ${stabilitySession.status}, not reconnecting`);
          }
        } else if (facesSession) {
          setOperationType('training_faces');
          // Only reconnect WebSocket for sessions with status 'running'
          if (facesSession.status === 'running') {
            setTaskId(facesSession.task_id);
            setOperationStarted(true);
            setCurrentStep(2);
            setStabilityComplete(true);

            // Restore progress from session
            if (facesSession.progress) {
              setCurrent(facesSession.progress.current || 0);
              setTotal(facesSession.progress.total || 0);
              setCurrentStatus(facesSession.progress.status || '');
              setCurrentItem(facesSession.progress.current_item || '');
            }

            setStatus('active');

            // Reconnect to WebSocket only for running sessions
            websocketService.connect('training_faces', facesSession.task_id, handleMessage, handleError);
          } else {
            // Session is cancelled, completed, or failed - don't reconnect
            console.log(`Session ${facesSession.task_id} has status ${facesSession.status}, not reconnecting`);
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

        // Handle image data for carousel (display for both stability and face training)
        if (data.progress.image_data && !isCarouselPaused) {
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
        // If stability training completed, mark it as complete
        if (operationType === 'training_stability') {
          setStabilityComplete(true);
        }
        break;

      case 'error':
        setCurrentStatus('Failed');
        setStatus('failed');
        window.alert(`${operationType === 'training_stability' ? 'Stability score' : 'Face detection'} training failed: ${data.error.message || 'Unknown error'}`);
        break;
      default:
        break;
    }
  };

  const handleError = (wsError) => {
    console.error('[Training] WebSocket error:', wsError);
    window.alert('WebSocket connection error. Progress updates may not be available.');
  };

  const startStabilityTraining = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setStatus('active');
      setOperationType('training_stability');
      const response = await apiService.startStabilityTraining({ source_dir: form.source_dir });
      setTaskId(response.task_id);
      setOperationStarted(true);
      setCurrentStep(1);

      websocketService.connect('training_stability', response.task_id, handleMessage, handleError);
    } catch (error) {
      console.error('Failed to start stability training:', error);
      window.alert('Failed to start stability training. Please check your configuration and try again.');
      setStatus('idle');
    } finally {
      setLoading(false);
    }
  };

  const startFaceDetectionTraining = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setStatus('active');
      setOperationType('training_faces');
      const response = await apiService.startFaceDetectionTraining(form);
      setTaskId(response.task_id);
      setOperationStarted(true);
      setCurrentStep(2);

      websocketService.connect('training_faces', response.task_id, handleMessage, handleError);
    } catch (error) {
      console.error('Failed to start face detection training:', error);
      window.alert('Failed to start face detection training. Please check your configuration and try again.');
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
    setOperationType('');
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

          {/* Step Indicator */}
          <div className="step-indicator">
            <div className={`step ${currentStep === 1 ? 'active' : ''} ${stabilityComplete ? 'complete' : ''}`}>
              <div className="step-number">
                {stabilityComplete ? '✓' : '1'}
              </div>
              <div className="step-label">Stability Scores</div>
            </div>
            <div className="step-divider"></div>
            <div className={`step ${currentStep === 2 ? 'active' : ''}`}>
              <div className="step-number">2</div>
              <div className="step-label">Face Detection</div>
            </div>
          </div>

          {/* Step 1: Stability Score Training */}
          {currentStep === 1 && (
            <form onSubmit={startStabilityTraining}>
              <div className="form-grid grid-1">
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
                  <p className="form-help">Directory containing images to process.</p>
                </div>
              </div>

              <div className="form-actions">
                <button type="submit" className="btn btn-primary btn-large" disabled={loading}>
                  <span className="btn-icon">🎯</span>
                  <span>{loading ? 'Starting...' : 'Start Stability Training'}</span>
                </button>
              </div>
            </form>
          )}

          {/* Step 2: Face Detection Training */}
          {currentStep === 2 && (
            <form onSubmit={startFaceDetectionTraining}>
              <div className="form-grid grid-1">
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
                  <p className="form-help">Directory containing images to process.</p>
                </div>
              </div>

              <div className="step-navigation">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setCurrentStep(1)}
                >
                  ← Back to Stability Scores
                </button>
                <button
                  type="submit"
                  className="btn btn-primary btn-large"
                  disabled={loading}
                >
                  <span className="btn-icon">🎯</span>
                  <span>{loading ? 'Starting...' : 'Start Face Detection Training'}</span>
                </button>
              </div>
            </form>
          )}

          {/* Step Navigation Buttons */}
          {currentStep === 1 && !stabilityComplete && (
            <div className="step-navigation">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setCurrentStep(2)}
              >
                Skip to Face Detection →
              </button>
            </div>
          )}
          {currentStep === 1 && stabilityComplete && (
            <div className="step-navigation">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => setCurrentStep(2)}
              >
                Proceed to Face Detection →
              </button>
            </div>
          )}
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
            operationType={operationType === 'training_stability' ? 'Stability Score Training' : 'Face Detection Training'}
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
            <strong>Training</strong> is a two-step process that processes your images for face detection and stability scoring.
          </p>

          <h4 className="info-subtitle">Step 1: Stability Scores</h4>
          <ul className="info-list">
            <li>✅ Calculates stability scores for each image</li>
            <li>✅ Classifies images (face/other)</li>
            <li>✅ Extracts content probability</li>
            <li>✅ Saves partial documents to MongoDB</li>
          </ul>

          <h4 className="info-subtitle">Step 2: Face Detection</h4>
          <ul className="info-list">
            <li>✅ Detects faces in images using InsightFace</li>
            <li>✅ Generates 512-dim embeddings</li>
            <li>✅ Extracts face metadata (age, gender, landmarks)</li>
            <li>✅ Merges with existing stability scores</li>
            <li>✅ Moves images without faces to noface directory</li>
            <li>✅ Real-time progress tracking</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Training;