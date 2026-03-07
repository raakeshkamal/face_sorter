import React, { useState, useMemo, useEffect } from 'react';
import './FolderPicker.css';
import { apiService } from '../services/api';

function FolderPicker({ show, fieldType = 'default', initialPath = '', onSelect, onCancel }) {
  const homePath = '/home';
  const [currentPath, setCurrentPath] = useState('');
  const [directories, setDirectories] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [hasParent, setHasParent] = useState(false);

  const pathSegments = useMemo(() => {
    if (currentPath === homePath || !currentPath) return [];
    const relativePath = currentPath.replace(homePath, '').replace(/^\//, '');
    return relativePath ? relativePath.split('/') : [];
  }, [currentPath]);

  const getLastUsedPath = () => {
    const key = `face_sorter_last_path_${fieldType}`;
    return localStorage.getItem(key) || '';
  };

  const saveLastUsedPath = (path) => {
    const key = `face_sorter_last_path_${fieldType}`;
    localStorage.setItem(key, path);
  };

  const loadDirectories = async (path) => {
    setLoading(true);
    setError('');
    try {
      const response = await apiService.browseDirectories(path);
      setDirectories(response.directories);
      setHasParent(response.has_parent);
      setCurrentPath(response.current_path);
    } catch (err) {
      console.error('Failed to load directories:', err);
      setError(err.response?.data?.detail || 'Failed to load directories');
      setDirectories([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (show) {
      const startingPath = initialPath || getLastUsedPath() || '';
      setCurrentPath(startingPath);
      loadDirectories(startingPath);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [show]);

  const navigateTo = (path) => {
    setCurrentPath(path);
    loadDirectories(path);
  };

  const navigateToParent = () => {
    const parentPath = currentPath.split('/').slice(0, -1).join('/') || '/';
    navigateTo(parentPath);
  };

  const navigateToPathSegment = (index) => {
    const segments = pathSegments.slice(0, index + 1);
    const newPath = [homePath, ...segments].join('/').replace('//', '/');
    navigateTo(newPath);
  };

  const handleSelect = () => {
    saveLastUsedPath(currentPath);
    onSelect && onSelect(currentPath);
  };

  const handleCancel = () => {
    onCancel && onCancel();
  };

  if (!show) return null;

  return (
    <div className="folder-picker-overlay" onClick={handleCancel}>
      <div className="folder-picker-modal" onClick={(e) => e.stopPropagation()}>
        <div className="folder-picker-header">
          <h3 className="folder-picker-title">Select Folder</h3>
          <button className="close-btn" onClick={handleCancel}>✕</button>
        </div>

        <div className="breadcrumb-container">
          <div className="breadcrumb">
            <button
              className={`breadcrumb-item ${currentPath === homePath ? 'active' : ''}`}
              onClick={() => navigateTo(homePath)}
            >
              🏠 Home
            </button>
            <span className="breadcrumb-separator">/</span>
            {currentPath !== homePath && pathSegments.map((segment, index) => (
              <React.Fragment key={index}>
                <button
                  className={`breadcrumb-item ${index === pathSegments.length - 1 ? 'active' : ''}`}
                  onClick={() => navigateToPathSegment(index)}
                >
                  {segment}
                </button>
                {index < pathSegments.length - 1 && (
                  <span className="breadcrumb-separator">/</span>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p className="loading-text">Loading directories...</p>
          </div>
        ) : error ? (
          <div className="error-container">
            <div className="error-icon">⚠️</div>
            <p className="error-text">{error}</p>
            <button className="btn btn-secondary" onClick={() => loadDirectories(currentPath)}>Retry</button>
          </div>
        ) : (
          <div className="directory-list">
            {hasParent && currentPath !== homePath && (
              <button className="directory-item parent-item" onClick={navigateToParent}>
                <span className="directory-icon">📁</span>
                <span className="directory-name">..</span>
                <span className="directory-path">Parent directory</span>
              </button>
            )}

            {directories.map((dir) => (
              <button key={dir.path} className="directory-item" onClick={() => navigateTo(dir.path)}>
                <span className="directory-icon">📁</span>
                <span className="directory-name">{dir.name}</span>
                <span className="directory-path">{dir.path}</span>
              </button>
            ))}

            {directories.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">📁</div>
                <p className="empty-text">No directories found</p>
              </div>
            )}
          </div>
        )}

        <div className="current-path-display">
          <span className="path-label">Current Path:</span>
          <span className="path-value">{currentPath}</span>
        </div>

        <div className="folder-picker-footer">
          <button className="btn btn-secondary" onClick={handleCancel}>Cancel</button>
          <button className="btn btn-primary" onClick={handleSelect}>Select</button>
        </div>
      </div>
    </div>
  );
}

export default FolderPicker;