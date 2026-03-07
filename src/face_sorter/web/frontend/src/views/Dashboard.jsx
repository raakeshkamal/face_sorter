import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './Dashboard.css';
import { apiService } from '../services/api';
import SkeletonLoader from '../components/SkeletonLoader.jsx';
import toastService from '../services/toast';

function Dashboard() {
  const [stats, setStats] = useState({
    total_faces: 0,
    total_classes: 0,
    total_clusters: 0,
    total_images: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadStats = async () => {
      try {
        setLoading(true);
        const data = await apiService.getOverview();
        setStats(data);
      } catch (error) {
        console.error('Failed to load statistics:', error);
        toastService.error('Failed to load statistics. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    loadStats();
  }, []);

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
              <span className="stat-icon">👥</span>
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

      <div className="quick-actions">
        <h2 className="section-title">Quick Actions</h2>
        <div className="actions-grid grid-2">
          <Link to="/training" className="card action-card">
            <span className="action-icon">🎯</span>
            <span className="action-title">Start Training</span>
            <p className="action-desc">Detect faces and generate embeddings</p>
          </Link>

          <Link to="/cleaning" className="card action-card">
            <span className="action-icon">🧹</span>
            <span className="action-title">Clean Dataset</span>
            <p className="action-desc">Validate and standardize images</p>
          </Link>

          <Link to="/sorting" className="card action-card">
            <span className="action-icon">🔀</span>
            <span className="action-title">Sort Faces</span>
            <p className="action-desc">Cluster and classify unknown faces</p>
          </Link>

          <Link to="/classes" className="card action-card">
            <span className="action-icon">👥</span>
            <span className="action-title">Manage Classes</span>
            <p className="action-desc">View and manage face classes</p>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;