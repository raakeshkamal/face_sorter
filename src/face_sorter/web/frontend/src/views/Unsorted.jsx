import React, { useState, useEffect } from 'react';
import './Unsorted.css';
import ImageGallery from '../components/ImageGallery.jsx';
import { apiService } from '../services/api';

function Unsorted() {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadUnsortedImages = async () => {
      try {
        setLoading(true);
        const data = await apiService.getUnsortedImages({ skip: 0, limit: 100 });
        setImages(data);
      } catch (error) {
        console.error('Failed to load unsorted images:', error);
      } finally {
        setLoading(false);
      }
    };
    loadUnsortedImages();
  }, []);

  const handleImageClick = (image) => {
    console.log('Image clicked:', image);
  };

  return (
    <div className="unsorted-view">
      <div className="view-header">
        <h1>📷 Unsorted</h1>
        <p className="subtitle">Browse and manage unsorted faces</p>
      </div>

      <ImageGallery
        images={images}
        loading={loading}
        onImageClick={handleImageClick}
      />

      {images.length > 0 && (
        <div className="unsorted-info">
          <p className="info-text">{images.length} unsorted faces found</p>
          <p className="info-hint">
            Sort these faces into classes or create new classes from clusters.
          </p>
        </div>
      )}
    </div>
  );
}

export default Unsorted;