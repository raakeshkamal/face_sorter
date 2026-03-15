import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './Dashboard.css';
import { apiService } from '../services/api';
import SkeletonLoader from '../components/SkeletonLoader.jsx';
import ImageGallery from '../components/ImageGallery.jsx';
import toastService from '../services/toast';

function Dashboard() {
  const [stats, setStats] = useState({
    total_faces: 0,
    total_classes: 0,
    total_clusters: 0,
    total_images: 0,
  });
  const [classes, setClasses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingClasses, setLoadingClasses] = useState(true);

  // Gallery state
  const [showGallery, setShowGallery] = useState(false);
  const [selectedClass, setSelectedClass] = useState(null);
  const [classImages, setClassImages] = useState([]);
  const [loadingImages, setLoadingImages] = useState(false);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const statsData = await apiService.getOverview();
        setStats(statsData);
      } catch (error) {
        console.error('Failed to load statistics:', error);
        toastService.error('Failed to load statistics.');
      } finally {
        setLoading(false);
      }

      try {
        setLoadingClasses(true);
        const classesData = await apiService.getClassSummaries();
        setClasses(classesData);
      } catch (error) {
        console.error('Failed to load classes:', error);
      } finally {
        setLoadingClasses(false);
      }
    };
    loadData();
  }, []);

  const getImageUrl = (face) => {
    if (face.cache_url) return `/images/${face.cache_url}`;
    return `/images/${face.filename || face.path}`;
  };

  const viewClass = async (cls) => {
    setSelectedClass(cls);
    setShowGallery(true);
    setLoadingImages(true);
    try {
      const images = await apiService.getClassImages(cls.class_name, { limit: 1000 });
      setClassImages(images);
    } catch (error) {
      console.error('Failed to fetch class images:', error);
      toastService.error('Failed to load class images.');
    } finally {
      setLoadingImages(false);
    }
  };

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Dashboard</h1>
        <p className="subtitle">Face Sorter Overview</p>
      </div>

      {loading ? (
        <div className="skeleton-wrapper">
          <SkeletonLoader type="stats" />
        </div>
      ) : (
        <div className="stats-grid grid-4">
          <div className="card stat-card">
            <div className="stat-header">
              <span className="stat-icon">👥</span>
              <span className="stat-title">Total Faces</span>
            </div>
            <div className="stat-value">{stats.total_faces}</div>
          </div>

          <div className="card stat-card">
            <div className="stat-header">
              <span className="stat-icon">🏷️</span>
              <span className="stat-title">Total Classes</span>
            </div>
            <div className="stat-value">{stats.total_classes}</div>
          </div>

          <div className="card stat-card">
            <div className="stat-header">
              <span className="stat-icon">🔀</span>
              <span className="stat-title">Total Clusters</span>
            </div>
            <div className="stat-value">{stats.total_clusters}</div>
          </div>

          <div className="card stat-card">
            <div className="stat-header">
              <span className="stat-icon">📷</span>
              <span className="stat-title">Total Images</span>
            </div>
            <div className="stat-value">{stats.total_images}</div>
          </div>
        </div>
      )}

      <div className="known-classes-section">
        <div className="section-header">
          <h2 className="section-title">Known Classes</h2>
          <Link to="/classes" className="view-all-link">Manage All Classes →</Link>
        </div>

        {loadingClasses ? (
          <div className="loading-placeholder">
            <div className="loading-spinner"></div>
            <p>Loading known classes...</p>
          </div>
        ) : classes.length > 0 ? (
          <div className="classes-grid">
            {classes.map((cls) => (
              <div key={cls.class_name} className="class-card card" onClick={() => viewClass(cls)}>
                <div className="class-header">
                  <span className="class-name">{cls.class_name}</span>
                  <span className="class-count">{cls.face_count} faces</span>
                </div>
                <div className="class-preview collage">
                  <div className="collage-main">
                    {cls.preview_faces[0] && (
                      <img
                        src={getImageUrl(cls.preview_faces[0])}
                        className="class-image main"
                        loading="lazy"
                        alt={`${cls.class_name} preview`}
                      />
                    )}
                  </div>
                  <div className="collage-side">
                    {cls.preview_faces.slice(1, 4).map((face, index) => (
                      <img
                        key={index}
                        src={getImageUrl(face)}
                        className="class-image side"
                        loading="lazy"
                        alt={`${cls.class_name} preview ${index + 1}`}
                      />
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-classes card">
            <div className="empty-icon">🏷️</div>
            <h3>No classes found</h3>
            <p>Go to the Sorting page to start classifying your face clusters.</p>
            <Link to="/sorting" className="btn btn-primary">Go to Sorting</Link>
          </div>
        )}
      </div>

      {showGallery && (
        <div className="modal-backdrop" onClick={() => setShowGallery(false)}>
          <div className="modal-content full-screen" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>{selectedClass?.class_name} Gallery ({selectedClass?.face_count} faces)</h2>
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

export default Dashboard;