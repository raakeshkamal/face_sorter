import React, { useState, useEffect } from 'react';
import './Classes.css';
import ImageGallery from '../components/ImageGallery.jsx';
import { apiService } from '../services/api';

function Classes() {
  const [classes, setClasses] = useState([]);
  const [loading, setLoading] = useState(true);

  // Gallery state
  const [showGallery, setShowGallery] = useState(false);
  const [selectedClass, setSelectedClass] = useState(null);
  const [classImages, setClassImages] = useState([]);
  const [loadingImages, setLoadingImages] = useState(false);

  const loadClasses = async () => {
    try {
      setLoading(true);
      const data = await apiService.getClassSummaries();
      setClasses(data);
    } catch (error) {
      console.error('Failed to load classes:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadClasses();
  }, []);

  const viewClass = async (classItem) => {
    setSelectedClass(classItem);
    setShowGallery(true);
    setLoadingImages(true);
    try {
      const images = await apiService.getClassImages(classItem.class_name, { limit: 1000 });
      setClassImages(images);
    } catch (error) {
      console.error('Failed to fetch class images:', error);
      window.alert('Failed to load class images.');
    } finally {
      setLoadingImages(false);
    }
  };

  const deleteClass = async (classItem) => {
    if (window.confirm(`Are you sure you want to delete class "${classItem.class_name}"?`)) {
      try {
        await apiService.deleteClass(classItem.class_name);
        loadClasses();
      } catch (error) {
        console.error('Failed to delete class:', error);
        window.alert('Failed to delete class');
      }
    }
  };

  const getImageUrl = (face) => {
    // Try full path first, then cache_url, then filename
    if (face.path) return `/images/${encodeURIComponent(face.path)}`;
    if (face.cache_url) return `/images/${encodeURIComponent(face.cache_url)}`;
    return `/images/${encodeURIComponent(face.filename || '')}`;
  };

  return (
    <div className="classes-view">
      <div className="view-header">
        <h1>👥 Classes</h1>
        <p className="subtitle">View and manage face classes</p>
      </div>

      {loading ? (
        <div className="loading">Loading classes...</div>
      ) : classes.length === 0 ? (
        <div className="no-classes">
          <span className="no-classes-icon">👥</span>
          <p className="no-classes-text">No classes found</p>
          <p className="no-classes-hint">
            Create classes from sorted clusters to organize your faces.
          </p>
        </div>
      ) : (
        <div className="classes-grid grid-3">
          {classes.map((classItem) => (
            <div key={classItem.class_name} className="card class-card">
              <div className="class-header">
                <div className="class-title">
                  <h3 className="class-name">{classItem.class_name}</h3>
                  <span className="class-count-badge">{classItem.face_count} faces</span>
                </div>
              </div>
              <div className="class-preview collage">
                {classItem.preview_faces.length > 0 ? (
                  <>
                    <div className="collage-main">
                      {classItem.preview_faces[0] && (
                        <img
                          src={getImageUrl(classItem.preview_faces[0])}
                          className="class-image main"
                          loading="lazy"
                          alt="Class main face"
                        />
                      )}
                    </div>
                    <div className="collage-side">
                      {classItem.preview_faces.slice(1, 4).map((face, index) => (
                        <img
                          key={index}
                          src={getImageUrl(face)}
                          className="class-image side"
                          loading="lazy"
                          alt="Class side face"
                        />
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="class-preview-empty">
                    <span className="empty-icon">📷</span>
                    <p>No preview images</p>
                  </div>
                )}
              </div>
              <div className="class-actions">
                <button className="btn btn-secondary" onClick={() => viewClass(classItem)}>
                  View
                </button>
                <button className="btn btn-secondary" onClick={() => deleteClass(classItem)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {showGallery && (
        <div className="modal-backdrop" onClick={() => setShowGallery(false)}>
          <div className="modal-content full-screen" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                <h2>Class: {selectedClass?.class_name} ({selectedClass?.face_count} faces)</h2>
              </div>
              <button className="modal-close" onClick={() => setShowGallery(false)}>✕</button>
            </div>
            <div className="modal-body">
              <ImageGallery 
                images={classImages} 
                loading={loadingImages} 
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Classes;