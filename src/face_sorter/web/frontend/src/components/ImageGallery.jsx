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
        <p className="no-images-text">No images found</p>
      </div>
    );
  }

  const selectedImage = selectedIndex !== null ? images[selectedIndex] : null;

  return (
    <div className="gallery-wrapper">
      {selectedIndex === null ? (
        <div className="gallery-grid-view">
          <div className="gallery-grid">
            {images.map((image, index) => (
              <div
                key={image.idx || image.filename || index}
                className="image-card"
                onClick={() => handleImageClick(image, index)}
              >
                <img
                  src={getImageUrl(image)}
                  alt={image.filename}
                  className="image-img"
                  loading="lazy"
                />
                {image.match_similarity !== null && image.match_similarity !== undefined ? (
                  <div className="badge-item match-badge">
                    {(image.match_similarity * 100).toFixed(1)}%
                  </div>
                ) : image.centroid_similarity !== null && image.centroid_similarity !== undefined ? (
                  <div className="badge-item cluster-badge">
                    {(image.centroid_similarity * 100).toFixed(1)}%
                  </div>
                ) : (
                  <div className="badge-item detection-badge">
                    {image.det_score?.toFixed(2) || 'N/A'}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="gallery-focus-view">
          <div className="focus-header">
            <button className="back-btn" onClick={closeDetail}>
              ← Back to Gallery
            </button>
            <div className="image-counter">
              Image {selectedIndex + 1} of {images.length}
            </div>
            <button className="close-btn-large" onClick={closeDetail}>✕</button>
          </div>

          <div className="focus-content">
            <button 
              className="nav-arrow prev" 
              onClick={goToPrev} 
              disabled={selectedIndex === 0}
            >
              ‹
            </button>

            <div className="focus-center-box">
              <div className="focus-image-container">
                <img
                  src={getImageUrl(selectedImage)}
                  alt={selectedImage.filename}
                  className="focus-image"
                />
              </div>

              <div className="focus-details-card">
                <div className="details-header">
                  <h3>Image Details</h3>
                  <div className="score-pills">
                    {selectedImage.match_similarity !== null && (
                      <span className="pill match">Match: {(selectedImage.match_similarity * 100).toFixed(1)}%</span>
                    )}
                    {selectedImage.centroid_similarity !== null && (
                      <span className="pill cluster">Cluster: {(selectedImage.centroid_similarity * 100).toFixed(1)}%</span>
                    )}
                    {!selectedImage.match_similarity && !selectedImage.centroid_similarity && (
                       <span className="pill detection">Confidence: {(selectedImage.det_score * 100).toFixed(1)}%</span>
                    )}
                  </div>
                </div>

                <div className="details-grid">
                  <div className="detail-item">
                    <span className="label">ID</span>
                    <span className="value">#{selectedImage.idx}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Age</span>
                    <span className="value">{selectedImage.age || 'Unknown'}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Gender</span>
                    <span className="value">{selectedImage.gender === 0 ? "Male" : selectedImage.gender === 1 ? "Female" : 'Unknown'}</span>
                  </div>
                  <div className="detail-item filename-item">
                    <span className="label">Filename</span>
                    <span className="value filename">{selectedImage.filename}</span>
                  </div>
                </div>

                {selectedImage.path && (
                  <div className="detail-item path-item">
                    <span className="label">System Path</span>
                    <span className="value path">{selectedImage.path}</span>
                  </div>
                )}
              </div>
            </div>

            <button 
              className="nav-arrow next" 
              onClick={goToNext} 
              disabled={selectedIndex === images.length - 1}
            >
              ›
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageGallery;
