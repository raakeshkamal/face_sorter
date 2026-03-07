<template>
  <div class="unsorted-view">
    <div class="view-header">
      <h1>📷 Unsorted</h1>
      <p class="subtitle">Browse and manage unsorted faces</p>
    </div>

    <ImageGallery
      :images="images"
      :loading="loading"
      @image-click="handleImageClick"
    />

    <div v-if="images.length > 0" class="unsorted-info">
      <p class="info-text">{{ images.length }} unsorted faces found</p>
      <p class="info-hint">
        Sort these faces into classes or create new classes from clusters.
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import ImageGallery from "../components/ImageGallery.vue";
import { apiService } from "../services/api";

const images = ref([]);
const loading = ref(true);

const loadUnsortedImages = async () => {
  try {
    loading.value = true;
    const skip = 0;
    const limit = 100;
    images.value = await apiService.getUnsortedImages({ skip, limit });
  } catch (error) {
    console.error("Failed to load unsorted images:", error);
  } finally {
    loading.value = false;
  }
};

const handleImageClick = (image) => {
  console.log("Image clicked:", image);
  // TODO: Implement image selection and class assignment
};

onMounted(() => {
  loadUnsortedImages();
});
</script>

<style scoped>
.unsorted-view {
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

.view-header {
  margin-bottom: 30px;
}

.view-header h1 {
  font-size: 36px;
  font-weight: 700;
  color: var(--secondary-color);
  margin: 0 0 10px 0;
}

.subtitle {
  font-size: 18px;
  color: #666;
  margin: 0;
}

.unsorted-info {
  text-align: center;
  padding: 20px;
  margin-top: 20px;
}

.info-text {
  font-size: 18px;
  font-weight: 600;
  color: var(--primary-color);
  margin: 0 0 10px 0;
}

.info-hint {
  color: #666;
  line-height: 1.6;
  margin: 0;
}
</style>