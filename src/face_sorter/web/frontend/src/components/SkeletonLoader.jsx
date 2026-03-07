import React from 'react';
import './SkeletonLoader.css';

function SkeletonLoader({ type = 'default', count = 4 }) {
  const renderContent = () => {
    if (type === 'card') {
      return (
        <div className="skeleton-card">
          <div className="skeleton-header">
            <div className="skeleton-title skeleton-shimmer"></div>
          </div>
          <div className="skeleton-body">
            <div className="skeleton-line skeleton-shimmer"></div>
            <div className="skeleton-line skeleton-shimmer"></div>
          </div>
        </div>
      );
    } else if (type === 'gallery') {
      return (
        <div className="skeleton-gallery grid-4">
          {Array.from({ length: count }).map((_, i) => (
            <div key={i} className="skeleton-image-card skeleton-shimmer"></div>
          ))}
        </div>
      );
    } else if (type === 'stats') {
      return (
        <div className="skeleton-stats grid-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="skeleton-stat-card skeleton-shimmer">
              <div className="skeleton-stat-icon"></div>
              <div className="skeleton-stat-value skeleton-shimmer"></div>
            </div>
          ))}
        </div>
      );
    } else if (type === 'list') {
      return (
        <div className="skeleton-list">
          {Array.from({ length: count }).map((_, i) => (
            <div key={i} className="skeleton-list-item">
              <div className="skeleton-avatar skeleton-shimmer"></div>
              <div className="skeleton-item-content">
                <div className="skeleton-item-title skeleton-shimmer"></div>
                <div className="skeleton-item-meta skeleton-shimmer"></div>
              </div>
            </div>
          ))}
        </div>
      );
    } else if (type === 'text') {
      return (
        <div className="skeleton-text">
          <div className="skeleton-text-line skeleton-shimmer"></div>
          <div className="skeleton-text-line skeleton-shimmer short"></div>
          <div className="skeleton-text-line skeleton-shimmer medium"></div>
        </div>
      );
    }

    return (
      <div className="skeleton-default">
        <div className="skeleton-block skeleton-shimmer"></div>
      </div>
    );
  };

  return <div className="skeleton-container">{renderContent()}</div>;
}

export default SkeletonLoader;