import React, { useState } from 'react';
import './Cleaning.css';
import ProgressBar from '../components/ProgressBar.jsx';
import FolderPicker from '../components/FolderPicker.jsx';
import { apiService } from '../services/api';

function Cleaning() {
  const [form, setForm] = useState({
    source_dir: '',
    output_dir: '',
    broken_dir: '',
    batch_size: 25,
    img_prefix: 'IMG',
    quality: 95,
    recursive: true,
    start_index: 0,
  });

  const [operationStarted, setOperationStarted] = useState(false);
  const [taskId, setTaskId] = useState('');
  const [loading, setLoading] = useState(false);

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
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const startCleaning = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      const response = await apiService.startCleaning(form);
      setTaskId(response.task_id);
      setOperationStarted(true);
    } catch (error) {
      console.error('Failed to start cleaning:', error);
      window.alert('Failed to start cleaning. Please check your configuration and try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    console.log('Cleaning cancelled');
    setOperationStarted(false);
    setTaskId('');
  };

  const handleReset = () => {
    setOperationStarted(false);
    setTaskId('');
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
                <p className="form-help">Directory containing images to clean</p>
              </div>

              <div className="form-group">
                <label className="form-label">Output Directory</label>
                <div className="input-with-browse">
                  <input
                    type="text"
                    name="output_dir"
                    value={form.output_dir}
                    onChange={handleChange}
                    className="form-input"
                    placeholder="/path/to/cleaned/images"
                    required
                  />
                  <button
                    type="button"
                    className="browse-btn"
                    onClick={() => openFolderPicker('output_dir')}
                  >
                    📁 Browse
                  </button>
                </div>
                <p className="form-help">Directory for cleaned images</p>
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
                    placeholder="/path/to/broken/images"
                  />
                  <button
                    type="button"
                    className="browse-btn"
                    onClick={() => openFolderPicker('broken_dir')}
                  >
                    📁 Browse
                  </button>
                </div>
                <p className="form-help">Directory for invalid/corrupted images</p>
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
            onCancel={handleCancel}
            onReset={handleReset}
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
            <li>✅ Real-time progress tracking</li>
            <li>✅ Supports various image formats (JPG, PNG, BMP, etc.)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Cleaning;