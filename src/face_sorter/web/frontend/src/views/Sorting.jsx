import React, { useState } from 'react';
import './Sorting.css';
import ProgressBar from '../components/ProgressBar.jsx';
import FolderPicker from '../components/FolderPicker.jsx';
import { apiService } from '../services/api';

function Sorting() {
  const [form, setForm] = useState({
    cache_dir: '',
    max_results: 10,
  });

  const [operationStarted, setOperationStarted] = useState(false);
  const [taskId, setTaskId] = useState('');
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [results, setResults] = useState({
    total_clusters: 0,
    total_faces: 0,
    assigned_classes: 0,
    clusters: [],
  });

  const [showFolderPicker, setShowFolderPicker] = useState(false);
  const [currentField, setCurrentField] = useState('');

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

  const showMockResults = () => {
    setOperationStarted(true);
    setShowResults(true);
    setResults({
      total_clusters: 8,
      total_faces: 156,
      assigned_classes: 12,
      clusters: [
        { id: 1, size: 24, faces: [] },
        { id: 2, size: 18, faces: [] },
        { id: 3, size: 15, faces: [] },
      ],
    });
  };

  const startSorting = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      // const response = await apiService.startSorting(form);
      // setTaskId(response.task_id);
      // setOperationStarted(true);
      showMockResults();
    } catch (error) {
      console.error('Failed to start sorting:', error);
      window.alert('Failed to start sorting. Please check your configuration and try again.');
    } finally {
      setLoading(false);
    }
  };

  const getImageUrl = (face) => {
    return `/images/${face.filename || face.path}`;
  };

  const viewCluster = (cluster) => {
    console.log('View cluster:', cluster);
  };

  const assignToClass = (cluster) => {
    console.log('Assign cluster to class:', cluster);
  };

  const handleCancel = () => {
    console.log('Sorting cancelled');
    setOperationStarted(false);
    setShowResults(false);
    setTaskId('');
  };

  const handleReset = () => {
    setOperationStarted(false);
    setShowResults(false);
    setTaskId('');
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
                <label className="form-label">Cache Directory</label>
                <div className="input-with-browse">
                  <input
                    type="text"
                    name="cache_dir"
                    value={form.cache_dir}
                    onChange={handleChange}
                    className="form-input"
                    placeholder="/path/to/cache"
                    required
                  />
                  <button
                    type="button"
                    className="browse-btn"
                    onClick={() => openFolderPicker('cache_dir')}
                  >
                    📁 Browse
                  </button>
                </div>
                <p className="form-help">Directory containing cached face embeddings</p>
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
                <p className="form-help">Number of top clusters to display (1-100)</p>
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
        <div className="progress-section">
          <ProgressBar
            operationType="Sorting"
            taskId={taskId}
            onCancel={handleCancel}
            onReset={handleReset}
          />
        </div>
      ) : null}

      {showResults && (
        <div className="results-section">
          <h2 className="results-title">Sorting Results</h2>
          <div className="results-stats grid-3">
            <div className="stat-card">
              <div className="stat-icon">👥</div>
              <div className="stat-value">{results.total_clusters}</div>
              <div className="stat-label">Clusters Found</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">📷</div>
              <div className="stat-value">{results.total_faces}</div>
              <div className="stat-label">Faces Sorted</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">👥</div>
              <div className="stat-value">{results.assigned_classes}</div>
              <div className="stat-label">Assigned to Classes</div>
            </div>
          </div>

          <div className="clusters-section">
            <h3 className="section-title">Top Clusters</h3>
            <div className="clusters-grid grid-4">
              {results.clusters.map((cluster) => (
                <div key={cluster.id} className="cluster-card card">
                  <div className="cluster-header">
                    <span className="cluster-id">Cluster #{cluster.id}</span>
                    <span className="cluster-size">{cluster.size} faces</span>
                  </div>
                  <div className="cluster-preview">
                    <div className="cluster-images">
                      {cluster.faces.slice(0, 4).map((face, index) => (
                        <img
                          key={index}
                          src={getImageUrl(face)}
                          className="cluster-image"
                          loading="lazy"
                          alt="Cluster face"
                        />
                      ))}
                    </div>
                  </div>
                  <div className="cluster-actions">
                    <button className="btn btn-secondary" onClick={() => viewCluster(cluster)}>
                      View Cluster
                    </button>
                    <button className="btn btn-primary" onClick={() => assignToClass(cluster)}>
                      Assign to Class
                    </button>
                  </div>
                </div>
              ))}
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