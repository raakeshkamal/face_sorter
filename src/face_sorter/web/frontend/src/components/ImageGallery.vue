<template>
  <div class="image-gallery">
    <div v-if="loading" class="loading">
      Loading images...
    </div>

    <div v-else-if="!images || images.length === 0" class="no-images">
      <span class="no-images-icon">📷</span>
      <p class="no-images-text">No images found</p>
    </div>

    <div v-else class="gallery-grid grid-4">
      <div
        v-for="image in images"
        :key="image.idx"
        class="image-card"
        @click="handleImageClick(image)"
      >
        <img
          :src="getImageUrl(image)"
          :alt="image.filename"
          class="image-img"
          loading="lazy"
        />
        <div class="image-overlay">
          <div class="image-info">
            <h4 class="image-title">{{ image.filename }}</h4>
            <p class="image-meta">
              Score: {{ image.det_score?.toFixed(3) }} | "N/A" }}
            </p>
            <p v-if="image.age" class="image-meta">
              Age: {{ image.age }}
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- Image Modal -->
    <div v-if="selectedImage" class="modal-backdrop" @click="closeModal">
      <div class="modal-content" @click.stop>
        <button class="modal-close" @click="closeModal">✕</button>
        <div class="modal-image-container">
          <img
            :src="getImageUrl(selectedImage)"
            :alt="selectedImage.filename"
            class="modal-image"
          />
        </div>
        <div class="modal-details">
          <h3>{{ selectedImage.filename }}</h3>
          <div class="detail-grid grid-2">
            <div class="detail-item">
              <span class="detail-label">Detection Score:</span>
              <span class="detail-value">{{ selectedImage.det_score?.toFixed(3) }}</span>
            </div>
            <div v-if="selectedImage.age" class="detail-item">
              <span class="detail-label">Age:</span>
              <span class="detail-value">{{ selectedImage.age }}</span>
            </div>
            <div v-if="selectedImage.gender !== undefined" class="detail-item">
              <span class="detail-label">Gender:</span>
              <span class="detail-value">{{ selectedImage.gender === 0 ? "Male" : "Female" }}</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">ID:</span>
              <span class="detail-value">{{ selectedImage.idx }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";

const props = defineProps({
  images: {
    type: Array,
    default: () => [],
  },
  loading: {
    type: Boolean,
    default: false,
  },
  cacheBaseUrl: {
    type: String,
    default: "/images",
  },
});

const emit = defineEmits(["imageClick"]);

const selectedImage = ref(null);

const getImageUrl = (image) => {
  if (image.cache_url) {
    return `${props.cacheBaseUrl}/${image.cache_url}`;
  }
  return `${props.cacheBaseUrl}/${image.filename}`;
};

const handleImageClick = (image) => {
  selectedImage.value = image;
  emit("imageClick", image);
};

const closeModal = () => {
  selectedImage.value = null;
};
</script>

<style scoped>
.image-gallery {
  padding: 20px 0;
  animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.no-images {
  text-align: center;
  padding: 60px 20px;
  color: #999;
}

.no-images-icon {
  font-size: 64px;
  display: block;
  margin-bottom: 20px;
}

.no-images-text {
  font-size: 18px;
  margin: 0;
}

/* Image Card Styling - fastdup inspired */
.image-card {
  position: relative;
  overflow: hidden;
  border-radius: 12px;
  background: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: pointer;
  aspect-ratio: 1;
}

.image-card:hover {
  transform: translateY(-8px) scale(1.02);
  box-shadow: 0 16px 40px rgba(101, 123, 236, 0.2);
}

.image-card:hover .image-overlay {
  opacity: 1;
}

.image-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  transition: transform 0.3s ease;
}

.image-card:hover .image-img {
  transform: scale(1.1);
}

.image-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: linear-gradient(
    to top,
    rgba(46, 62, 142, 0.95) 0%,
    rgba(46, 62, 142, 0.85) 100%
  );
  color: white;
  padding: 20px 15px 15px;
  opacity: 0;
  transition: opacity 0.3s ease;
  backdrop-filter: blur(4px);
}

.image-info {
  transform: translateY(0);
  transition: transform 0.3s ease;
}

.image-title {
  font-size: 14px;
  font-weight: 600;
  margin: 0 0 8px 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.image-meta {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.9);
  margin: 0;
  line-height: 1.3;
}

/* Modal Styling */
.modal-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: modalFadeIn 0.3s ease;
}

@keyframes modalFadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.modal-content {
  background: white;
  border-radius: 16px;
  max-width: 90vw;
  max-height: 90vh;
  overflow-y: auto;
  position: relative;
  animation: modalSlideUp 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
}

@keyframes modalSlideUp {
  from {
    transform: translateY(50px) scale(0.9);
    opacity: 0;
  }
  to {
    transform: translateY(0) scale(1);
    opacity: 1;
  }
}

.modal-close {
  position: absolute;
  top: 15px;
  right: 15px;
  background: var(--primary-color);
  color: white;
  border: none;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  font-size: 24px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
}

.modal-close:hover {
  background: var(--secondary-color);
  transform: rotate(90deg);
}

.modal-image-container {
  text-align: center;
  padding: 20px;
  background: #f8f9fa;
}

.modal-image {
  max-width: 100%;
  max-height: 70vh;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.modal-details {
  padding: 24px 30px;
  background: white;
  border-top: 1px solid #e0e0e0;
}

.modal-details h3 {
  font-size: 20px;
  font-weight: 600;
  color: var(--secondary-color);
  margin: 0 0 20px 0;
}

.detail-grid {
  display: grid;
  gap: 16px;
}

.detail-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.detail-label {
  font-size: 12px;
  font-weight: 600;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.detail-value {
  font-size: 16px;
  font-weight: 500;
  color: var(--text-color);
}

/* Responsive Design */
@media (max-width: 768px) {
  .gallery-grid {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  }

  .modal-content {
    width: 95vw;
    max-height: 85vh;
  }

  .modal-details {
    padding: 20px;
  }

  .detail-grid {
    grid-template-columns: 1fr;
  }
}
</style>