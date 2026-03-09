import React, { useState, useEffect, useCallback } from 'react';
import './ImageGallery.css';

function ImageGallery({ images = [], loading = false, cacheBaseUrl = '/images', onImageClick }) {
  const [selectedIndex, setSelectedIndex] = useState(null);

  const getImageUrl = useCallback((image) => {
    if (!image) return '';
    if (image.cache_url) {
      return `${cacheBaseUrl}/${image.cache_url}`;
    }
    return `${cacheBaseUrl}/${image.filename}`;
  }, [cacheBaseUrl]);

  const handleImageClick = (image, index) => {
    setSelectedIndex(index);
    if (onImageClick) {
      onImageClick(image);
    }
  };

  const closeDetail = () => {
    setSelectedIndex(null);
  };

  const goToNext = useCallback(() => {
    if (selectedIndex !== null && selectedIndex < images.length - 1) {
      setSelectedIndex(selectedIndex + 1);
    }
  }, [selectedIndex, images.length]);

  const goToPrev = useCallback(() => {
    if (selectedIndex !== null && selectedIndex > 0) {
      setSelectedIndex(selectedIndex - 1);
    }
  }, [selectedIndex]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (selectedIndex === null) return;

      if (e.key === 'ArrowRight') {
        goToNext();
      } else if (e.key === 'ArrowLeft') {
        goToPrev();
      } else if (e.key === 'Escape') {
        closeDetail();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedIndex, goToNext, goToPrev]);

  if (loading) {
    return (
      <div className="image-gallery loading-state">
        <div className="loading-spinner"></div>
        <p>Loading gallery...</p>
      </div>
    );
  }

  if (!images || images.length === 0) {
    return (
      <div className="no-images">
        <span className="no-images-icon">📷</span>
        <p className="no-images-text">No images found in this cluster</p>
      </div>
    );
  }

  const selectedImage = selectedIndex !== null ? images[selectedIndex] : null;

  return (
    <div className={`gallery-container ${selectedIndex !== null ? 'split-view' : 'grid-view'}`}>
      <div className="gallery-main">
        <div className="gallery-grid">
          {images.map((image, index) => (
            <div
              key={image.idx || image.filename || index}
              className={`image-card ${selectedIndex === index ? 'selected' : ''}`}
              onClick={() => handleImageClick(image, index)}
            >
              <img
                src={getImageUrl(image)}
                alt={image.filename}
                className="image-img"
                loading="lazy"
              />
              <div className="image-badge">
                {image.det_score?.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      </div>

      {selectedImage && (
        <div className="gallery-side-panel">
          <div className="panel-header">
            <h3>Image Details</h3>
            <button className="panel-close" onClick={closeDetail}>✕</button>
          </div>
          
          <div className="panel-content">
            <div className="detail-image-container">
              <img
                src={getImageUrl(selectedImage)}
                alt={selectedImage.filename}
                className="detail-image"
              />
              
              <div className="panel-navigation">
                <button 
                  className="nav-btn prev" 
                  onClick={goToPrev} 
                  disabled={selectedIndex === 0}
                >
                  ‹
                </button>
                <button 
                  className="nav-btn next" 
                  onClick={goToNext} 
                  disabled={selectedIndex === images.length - 1}
                >
                  ›
                </button>
              </div>
            </div>

            <div className="detail-info">
              <div className="info-row">
                <span className="info-label">Filename</span>
                <span className="info-value filename">{selectedImage.filename}</span>
              </div>
              <div className="info-grid">
                <div className="info-row">
                  <span className="info-label">Score</span>
                  <span className="info-value">{selectedImage.det_score?.toFixed(3) || 'N/A'}</span>
                </div>
                {selectedImage.age !== undefined && (
                  <div className="info-row">
                    <span className="info-label">Age</span>
                    <span className="info-value">{selectedImage.age}</span>
                  </div>
                )}
                {selectedImage.gender !== undefined && (
                  <div className="info-row">
                    <span className="info-label">Gender</span>
                    <span className="info-value">{selectedImage.gender === 0 ? "Male" : "Female"}</span>
                  </div>
                )}
                {selectedImage.idx !== undefined && (
                  <div className="info-row">
                    <span className="info-label">ID</span>
                    <span className="info-value">#{selectedImage.idx}</span>
                  </div>
                )}
              </div>
              
              {selectedImage.path && (
                <div className="info-row full-width">
                  <span className="info-label">Full Path</span>
                  <span className="info-value path-text" title={selectedImage.path}>
                    {selectedImage.path}
                  </span>
                </div>
              )}
            </div>
          </div>
          
          <div className="panel-footer">
            <div className="image-counter">
              Image {selectedIndex + 1} of {images.length}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageGallery;
