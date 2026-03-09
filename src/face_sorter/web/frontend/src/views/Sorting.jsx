import React, { useState, useEffect } from 'react';
import './Sorting.css';
import ProgressBar from '../components/ProgressBar.jsx';
import FolderPicker from '../components/FolderPicker.jsx';
import ImageGallery from '../components/ImageGallery.jsx';
import { apiService } from '../services/api';
import websocketService from '../services/websocket';

function Sorting() {
  const [form, setForm] = useState({
    source_dir: '',
    max_results: 10,
    min_samples: 2,
    min_cluster_size: 5,
  });

  const [operationStarted, setOperationStarted] = useState(false);
  const [taskId, setTaskId] = useState('');
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [results, setResults] = useState({
    total_clusters: 0,
    total_faces: 0,
    assigned_classes: 0,
    clusters: [],
  });

  // Simplified state for removing progress bar
  const [status, setStatus] = useState('idle');

  // Gallery state
  const [showGallery, setShowGallery] = useState(false);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [clusterImages, setClusterImages] = useState([]);
  const [loadingImages, setLoadingImages] = useState(false);

  const [showFolderPicker, setShowFolderPicker] = useState(false);
  const [currentField, setCurrentField] = useState('');

  // Check for active sessions on mount
  useEffect(() => {
    const checkActiveSession = async () => {
      try {
        const activeSessions = await apiService.getActiveSessions();
        const sortingSession = activeSessions.find(s => s.operation_type === 'sorting');
        if (sortingSession && sortingSession.status === 'running') {
          setTaskId(sortingSession.task_id);
          setOperationStarted(true);
          setStatus('active');
          websocketService.connect('sorting', sortingSession.task_id, handleMessage, handleError);
        }
      } catch (error) {
        console.error('Failed to check active sessions:', error);
      }
    };
    checkActiveSession();
    return () => websocketService.disconnect();
  }, []);

  const handleMessage = (data) => {
    switch (data.type) {
      case 'progress':
        if (data.progress.status === 'Complete') {
          setStatus('complete');
          fetchResults();
        }
        break;
      case 'complete':
        setStatus('complete');
        fetchResults();
        break;
      case 'error':
        setStatus('failed');
        window.alert(`Sorting failed: ${data.error.message || 'Unknown error'}`);
        break;
      default:
        break;
    }
  };

  const handleError = (error) => {
    console.error('WebSocket error:', error);
  };

  const fetchResults = async () => {
    try {
      setLoading(true);
      const clusters = await apiService.getClusters(form.max_results);
      const overview = await apiService.getOverview();
      
      setResults({
        total_clusters: clusters.length,
        total_faces: overview.total_faces,
        assigned_classes: overview.total_classes,
        clusters: clusters.map(c => ({
          id: c.cluster_id,
          size: c.size,
          faces: c.preview_faces
        })),
      });
      setShowResults(true);
    } catch (error) {
      console.error('Failed to fetch results:', error);
    } finally {
      setLoading(false);
    }
  };

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

  const startSorting = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      const response = await apiService.startSorting(form);
      setTaskId(response.task_id);
      setOperationStarted(true);
      setStatus('active');
      websocketService.connect('sorting', response.task_id, handleMessage, handleError);
    } catch (error) {
      console.error('Failed to start sorting:', error);
      window.alert('Failed to start sorting. Please check your configuration and try again.');
    } finally {
      setLoading(false);
    }
  };

  const getImageUrl = (face) => {
    if (face.cache_url) return `/images/${face.cache_url}`;
    return `/images/${face.filename || face.path}`;
  };

  const viewCluster = async (cluster) => {
    setSelectedCluster(cluster);
    setShowGallery(true);
    setLoadingImages(true);
    try {
      const images = await apiService.getClusterImages(cluster.id, { limit: 1000 });
      setClusterImages(images);
    } catch (error) {
      console.error('Failed to fetch cluster images:', error);
      window.alert('Failed to load cluster images.');
    } finally {
      setLoadingImages(false);
    }
  };

  const assignToClass = async (cluster) => {
    const className = window.prompt(`Enter class name for Cluster #${cluster.id}:`);
    if (!className) return;

    try {
      setLoading(true);
      await apiService.createClass({
        class_name: className,
        cluster_id: cluster.id
      });
      window.alert(`Successfully assigned Cluster #${cluster.id} to class "${className}"`);
      // Refresh results to show updated class count
      fetchResults();
    } catch (error) {
      console.error('Failed to assign class:', error);
      window.alert(`Failed to assign class: ${error.response?.data?.detail || error.message}`);
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
      setOperationStarted(false);
    } catch (error) {
      console.error('Failed to cancel:', error);
    } finally {
      setCancelling(false);
    }
  };

  const handleReset = () => {
    setOperationStarted(false);
    setShowResults(false);
    setTaskId('');
    setStatus('idle');
    setResults({
      total_clusters: 0,
      total_faces: 0,
      assigned_classes: 0,
      clusters: [],
    });
  };

  return (
    <div className="sorting-view">
      <div className="view-header">
        <h1>🔀 Sorting</h1>
        <p className="subtitle">Cluster and classify unknown faces</p>
      </div>

      {!operationStarted ? (
        <div className="sorting-config card">
          <h2 className="config-title">Sorting Configuration</h2>
          <form onSubmit={startSorting}>
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
                  />
                  <button
                    type="button"
                    className="browse-btn"
                    onClick={() => openFolderPicker('source_dir')}
                  >
                    📁 Browse
                  </button>
                </div>
                <p className="form-help">Directory containing images. The system will automatically find the cache sibling folder.</p>
              </div>

              <div className="form-group">
                <label className="form-label">Max Results</label>
                <input
                  type="number"
                  name="max_results"
                  value={form.max_results}
                  onChange={handleChange}
                  className="form-input"
                  min="1"
                  max="100"
                  placeholder="10"
                />
                <p className="form-help">Maximum number of clusters to show in preview.</p>
              </div>

              <div className="form-group">
                <label className="form-label">Min Cluster Size</label>
                <input
                  type="number"
                  name="min_cluster_size"
                  value={form.min_cluster_size}
                  onChange={handleChange}
                  className="form-input"
                  min="2"
                  max="100"
                  placeholder="5"
                />
                <p className="form-help">Minimum number of faces to form a cluster.</p>
              </div>

              <div className="form-group">
                <label className="form-label">Min Samples</label>
                <input
                  type="number"
                  name="min_samples"
                  value={form.min_samples}
                  onChange={handleChange}
                  className="form-input"
                  min="1"
                  max="50"
                  placeholder="2"
                />
                <p className="form-help">Lower values create more (potentially noisy) clusters.</p>
              </div>
              </div>

            <div className="info-box">
              <p className="info-text">
                <strong>Sorting</strong> will cluster unknown faces using HDBSCAN and display
                the top clusters for manual classification.
              </p>
              <ul className="info-features">
                <li>✅ Automatic face clustering</li>
                <li>✅ Display top clusters by similarity</li>
                <li>✅ Quick class assignment from clusters</li>
                <li>✅ Real-time progress tracking</li>
              </ul>
            </div>

            <div className="form-actions">
              <button type="submit" className="btn btn-primary btn-large" disabled={loading}>
                <span className="btn-icon">🔀</span>
                <span>{loading ? 'Starting...' : 'Start Sorting'}</span>
              </button>
            </div>
          </form>
        </div>
      ) : !showResults ? (
        <div className="sorting-loading card">
          <div className="loading-content">
            <div className="loading-spinner-large"></div>
            <h2>Analyzing & Clustering Faces</h2>
            <p>We are processing your embeddings to find similar faces. This can take several minutes for large datasets (30k+ images).</p>
            <div className="loading-status">Status: <span>Working...</span></div>
            <button className="btn btn-secondary" onClick={handleCancel} disabled={cancelling}>
              {cancelling ? 'Stopping...' : 'Cancel Operation'}
            </button>
          </div>
        </div>
      ) : null}

      {showResults && (
        <div className="results-section">
          <h2 className="results-title">Sorting Results</h2>
          <div className="results-stats">
            <div className="stat-card card">
              <div className="stat-icon">👥</div>
              <div className="stat-value">{results.total_clusters}</div>
              <div className="stat-label">Clusters Found</div>
            </div>
            <div className="stat-card card">
              <div className="stat-icon">📷</div>
              <div className="stat-value">{results.total_faces}</div>
              <div className="stat-label">Faces in Database</div>
            </div>
            <div className="stat-card card">
              <div className="stat-icon">🏷️</div>
              <div className="stat-value">{results.assigned_classes}</div>
              <div className="stat-label">Known Classes</div>
            </div>
          </div>

          <div className="clusters-section">
            <h3 className="section-title">Top Clusters</h3>
            <div className="clusters-grid">
              {results.clusters.map((cluster) => (
                <div key={cluster.id} className="cluster-card card">
                  <div className="cluster-header">
                    <span className="cluster-id">Cluster #{cluster.id}</span>
                    <span className="cluster-size">{cluster.size} faces</span>
                  </div>
                  <div className="cluster-preview collage">
                    <div className="collage-main">
                      {cluster.faces[0] && (
                        <img
                          src={getImageUrl(cluster.faces[0])}
                          className="cluster-image main"
                          loading="lazy"
                          alt="Cluster main face"
                        />
                      )}
                    </div>
                    <div className="collage-side">
                      {cluster.faces.slice(1, 4).map((face, index) => (
                        <img
                          key={index}
                          src={getImageUrl(face)}
                          className="cluster-image side"
                          loading="lazy"
                          alt="Cluster side face"
                        />
                      ))}
                    </div>
                  </div>
                  <div className="cluster-actions">
                    <button className="btn btn-secondary" onClick={() => viewCluster(cluster)}>
                      🔍 View
                    </button>
                    <button className="btn btn-primary" onClick={() => assignToClass(cluster)}>
                      🏷️ Assign
                    </button>
                  </div>
                </div>
              ))}
            </div>
            <div className="form-actions">
               <button className="btn btn-secondary" onClick={handleReset}>
                  Start New Sorting
               </button>
            </div>
          </div>
        </div>
      )}

      {showGallery && (
        <div className="modal-backdrop" onClick={() => setShowGallery(false)}>
          <div className="modal-content full-screen" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Cluster #{selectedCluster?.id} Gallery ({selectedCluster?.size} faces)</h2>
              <button className="modal-close" onClick={() => setShowGallery(false)}>✕</button>
            </div>
            <div className="modal-body">
              <ImageGallery 
                images={clusterImages} 
                loading={loadingImages} 
              />
            </div>
            <div className="modal-footer">
               <button className="btn btn-primary" onClick={() => assignToClass(selectedCluster)}>
                  Assign Cluster to Class
               </button>
            </div>
          </div>
        </div>
      )}

      <FolderPicker
        show={showFolderPicker}
        fieldType={currentField}
        initialPath={form[currentField] || ''}
        onSelect={handleFolderSelect}
        onCancel={handleFolderCancel}
      />
    </div>
  );
}

export default Sorting;
