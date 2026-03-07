import React, { useState } from 'react';
import './ImageGallery.css';

function ImageGallery({ images = [], loading = false, cacheBaseUrl = '/images', onImageClick }) {
  const [selectedImage, setSelectedImage] = useState(null);

  const getImageUrl = (image) => {
    if (image.cache_url) {
      return `${cacheBaseUrl}/${image.cache_url}`;
    }
    return `${cacheBaseUrl}/${image.filename}`;
  };

  const handleImageClick = (image) => {
    setSelectedImage(image);
    if (onImageClick) {
      onImageClick(image);
    }
  };

  const closeModal = () => {
    setSelectedImage(null);
  };

  return (
    <div className="image-gallery">
      {loading ? (
        <div className="loading">Loading images...</div>
      ) : !images || images.length === 0 ? (
        <div className="no-images">
          <span className="no-images-icon">📷</span>
          <p className="no-images-text">No images found</p>
        </div>
      ) : (
        <div className="gallery-grid grid-4">
          {images.map((image) => (
            <div
              key={image.idx || image.filename}
              className="image-card"
              onClick={() => handleImageClick(image)}
            >
              <img
                src={getImageUrl(image)}
                alt={image.filename}
                className="image-img"
                loading="lazy"
              />
              <div className="image-overlay">
                <div className="image-info">
                  <h4 className="image-title">{image.filename}</h4>
                  <p className="image-meta">
                    Score: {image.det_score?.toFixed(3) || 'N/A'}
                  </p>
                  {image.age !== undefined && (
                    <p className="image-meta">Age: {image.age}</p>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedImage && (
        <div className="modal-backdrop" onClick={closeModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeModal}>✕</button>
            <div className="modal-image-container">
              <img
                src={getImageUrl(selectedImage)}
                alt={selectedImage.filename}
                className="modal-image"
              />
            </div>
            <div className="modal-details">
              <h3>{selectedImage.filename}</h3>
              <div className="detail-grid grid-2">
                <div className="detail-item">
                  <span className="detail-label">Detection Score:</span>
                  <span className="detail-value">{selectedImage.det_score?.toFixed(3) || 'N/A'}</span>
                </div>
                {selectedImage.age !== undefined && (
                  <div className="detail-item">
                    <span className="detail-label">Age:</span>
                    <span className="detail-value">{selectedImage.age}</span>
                  </div>
                )}
                {selectedImage.gender !== undefined && (
                  <div className="detail-item">
                    <span className="detail-label">Gender:</span>
                    <span className="detail-value">{selectedImage.gender === 0 ? "Male" : "Female"}</span>
                  </div>
                )}
                {selectedImage.idx !== undefined && (
                  <div className="detail-item">
                    <span className="detail-label">ID:</span>
                    <span className="detail-value">{selectedImage.idx}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageGallery;