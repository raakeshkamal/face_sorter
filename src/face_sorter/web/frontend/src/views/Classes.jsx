import React, { useState, useEffect } from 'react';
import './Classes.css';
import { apiService } from '../services/api';

function Classes() {
  const [classes, setClasses] = useState([]);
  const [loading, setLoading] = useState(true);

  const loadClasses = async () => {
    try {
      setLoading(true);
      const data = await apiService.getClasses();
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

  const viewClass = (classItem) => {
    console.log('View class:', classItem.class_name);
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
                <span className="class-icon">👥</span>
                <h3 className="class-name">{classItem.class_name}</h3>
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
    </div>
  );
}

export default Classes;