import React, { useState, useEffect, useRef, useCallback } from 'react';
import './ImageCarousel.css';

function ImageCarousel({
  images = [],
  currentIndex = 0,
  totalImages = 0,
  autoAdvance = true,
  onPauseToggle = null,
  className = ''
}) {
  const [displayIndex, setDisplayIndex] = useState(currentIndex);
  const [isPaused, setIsPaused] = useState(false);
  const [imageStates, setImageStates] = useState({});
  const carouselRef = useRef(null);
  const autoAdvanceRef = useRef(autoAdvance);
  const imagesRef = useRef(images);
  const currentIndexRef = useRef(currentIndex);

  // Helper function to construct image URLs (consistent with ImageGallery pattern)
  const getImageUrl = useCallback((image) => {
    if (image && image.cache_url) {
      return `/images/${image.cache_url}`;
    }
    if (image && image.filename) {
      return `/images/${image.filename}`;
    }
    return '';
  }, []);

  // Update refs when props change
  useEffect(() => {
    autoAdvanceRef.current = autoAdvance;
  }, [autoAdvance]);

  useEffect(() => {
    imagesRef.current = images;
  }, [images]);

  useEffect(() => {
    currentIndexRef.current = currentIndex;
  }, [currentIndex]);

  // Handle image loading states using filename as key
  const handleImageLoad = useCallback((key) => {
    if (!key) return;
    setImageStates(prev => ({
      ...prev,
      [key]: { ...prev[key], loaded: true, loading: false, error: false }
    }));
  }, []);

  const handleImageError = useCallback((key) => {
    if (!key) return;
    console.error(`[ImageCarousel] Failed to load image: ${key}`);
    setImageStates(prev => ({
      ...prev,
      [key]: { ...prev[key], loaded: false, loading: false, error: true }
    }));
  }, []);

  const handleImageLoadStart = useCallback((key) => {
    if (!key) return;
    setImageStates(prev => ({
      ...prev,
      [key]: { ...prev[key], loading: true, error: false }
    }));
  }, []);

  // Auto-advance to new images
  useEffect(() => {
    if (currentIndex !== displayIndex && autoAdvanceRef.current && !isPaused) {
      setDisplayIndex(currentIndex);
    }
  }, [currentIndex, displayIndex, isPaused]);

  // Preload nearby images for smooth transitions
  useEffect(() => {
    const preloadImages = (indices) => {
      indices.forEach((idx) => {
        if (idx >= 0 && idx < images.length) {
          const image = images[idx];
          const key = image?.filename;
          if (key && !imageStates[key]?.preloaded && !imageStates[key]?.loaded) {
            const img = new Image();
            const imageUrl = getImageUrl(image);
            if (imageUrl) {
              img.src = imageUrl;
              img.onload = () => {
                setImageStates(prev => ({
                  ...prev,
                  [key]: { ...prev[key], preloaded: true, loaded: true }
                }));
              };
              img.onerror = () => {
                setImageStates(prev => ({
                  ...prev,
                  [key]: { ...prev[key], error: true }
                }));
              };
            }
          }
        }
      });
    };

    // Preload current, next, and previous images
    const indicesToPreload = [
      displayIndex,
      displayIndex + 1,
      displayIndex - 1
    ];
    preloadImages(indicesToPreload);
  }, [displayIndex, images, imageStates, getImageUrl]);

  // Handle pause on hover
  const handleMouseEnter = () => {
    setIsPaused(true);
    if (onPauseToggle) {
      onPauseToggle(true);
    }
  };

  const handleMouseLeave = () => {
    setIsPaused(false);
    if (onPauseToggle) {
      onPauseToggle(false);
    }
  };

  // Manual navigation
  const goToPrevious = () => {
    if (displayIndex > 0) {
      setDisplayIndex(displayIndex - 1);
    }
  };

  const goToNext = () => {
    if (displayIndex < images.length - 1) {
      setDisplayIndex(displayIndex + 1);
    }
  };

  const goToImage = (index) => {
    if (index >= 0 && index < images.length) {
      setDisplayIndex(index);
    }
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowLeft') {
        goToPrevious();
      } else if (e.key === 'ArrowRight') {
        goToNext();
      } else if (e.key === ' ') {
        e.preventDefault();
        setIsPaused(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [displayIndex, images.length]);

  // Cleanup old image states to prevent memory leaks
  useEffect(() => {
    if (images.length === 0) return;

    // Keep only states for images currently in the images array
    const currentFilenames = new Set(images.map(img => img.filename).filter(Boolean));

    setImageStates(prev => {
      let hasChanged = false;
      const newState = {};
      
      Object.keys(prev).forEach(key => {
        if (currentFilenames.has(key)) {
          newState[key] = prev[key];
        } else {
          hasChanged = true;
        }
      });
      
      return hasChanged ? newState : prev;
    });
  }, [images]);

  const currentImage = images[displayIndex];
  const hasImages = images.length > 0;
  const imageKey = currentImage?.filename || `idx-${displayIndex}`;

  if (!hasImages) {
    return (
      <div className={`image-carousel ${className}`}>
        <div className="carousel-empty">
          <div className="empty-icon">📷</div>
          <p className="empty-text">Waiting for images...</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`image-carousel ${isPaused ? 'paused' : ''} ${className}`}
      ref={carouselRef}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      role="region"
      aria-label="Image carousel"
    >
      {/* Progress overlay */}
      <div className="carousel-progress">
        <span className="progress-text">
          {displayIndex + 1} / {totalImages || images.length}
        </span>
        {isPaused && <span className="pause-indicator">⏸ Paused</span>}
      </div>

      {/* Main image display */}
      <div className="carousel-viewport">
        <div className="carousel-track">
          {currentImage && (
            <div className="carousel-slide active">
              <div className="image-container">
                {imageStates[imageKey]?.error ? (
                  <div className="image-error">
                    <span className="error-icon">⚠️</span>
                    <span className="error-text">Failed to load image</span>
                  </div>
                ) : (
                  <img
                    src={getImageUrl(currentImage)}
                    alt={currentImage.filename || `Image ${displayIndex + 1}`}
                    className="carousel-image"
                    loaded={imageStates[imageKey]?.loaded ? "true" : "false"}
                    onLoad={() => handleImageLoad(imageKey)}
                    onError={() => handleImageError(imageKey)}
                    onLoadStart={() => handleImageLoadStart(imageKey)}
                    loading="lazy"
                  />
                )}
                {imageStates[imageKey]?.loading && !imageStates[imageKey]?.loaded && (
                  <div className="image-loading">
                    <div className="loading-spinner"></div>
                  </div>
                )}
              </div>

              {/* Metadata overlay */}
              <div className="image-metadata">
                {currentImage.filename && (
                  <div className="metadata-item">
                    <span className="metadata-label">Filename:</span>
                    <span className="metadata-value">{currentImage.filename}</span>
                  </div>
                )}
                {currentImage.det_score !== undefined && (
                  <div className="metadata-item">
                    <span className="metadata-label">Detection Score:</span>
                    <span className="metadata-value">{currentImage.det_score.toFixed(3)}</span>
                  </div>
                )}
                {currentImage.age !== undefined && (
                  <div className="metadata-item">
                    <span className="metadata-label">Age:</span>
                    <span className="metadata-value">{currentImage.age}</span>
                  </div>
                )}
                {currentImage.gender !== undefined && (
                  <div className="metadata-item">
                    <span className="metadata-label">Gender:</span>
                    <span className="metadata-value">
                      {currentImage.gender === 0 ? 'Male' : 'Female'}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Navigation controls */}
      {images.length > 1 && (
        <div className="carousel-controls">
          <button
            className="carousel-nav-btn prev"
            onClick={goToPrevious}
            disabled={displayIndex === 0}
            aria-label="Previous image"
            title="Previous image (←)"
          >
            ‹
          </button>

          <div className="carousel-indicators">
            {images.slice(0, 10).map((_, idx) => (
              <button
                key={idx}
                className={`indicator ${idx === displayIndex ? 'active' : ''}`}
                onClick={() => goToImage(idx)}
                aria-label={`Go to image ${idx + 1}`}
                title={`Image ${idx + 1}`}
              />
            ))}
            {images.length > 10 && (
              <span className="indicator-ellipsis">...</span>
            )}
          </div>

          <button
            className="carousel-nav-btn next"
            onClick={goToNext}
            disabled={displayIndex === images.length - 1}
            aria-label="Next image"
            title="Next image (→)"
          >
            ›
          </button>
        </div>
      )}

      {/* Keyboard hint */}
      <div className="keyboard-hint">
        <span className="hint-text">Use arrow keys to navigate, space to pause</span>
      </div>
    </div>
  );
}

export default ImageCarousel;
