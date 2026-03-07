<template>
  <div class="skeleton-container">
    <div v-if="type === 'card'" class="skeleton-card">
      <div class="skeleton-header">
        <div class="skeleton-title skeleton-shimmer"></div>
      </div>
      <div class="skeleton-body">
        <div class="skeleton-line skeleton-shimmer"></div>
        <div class="skeleton-line skeleton-shimmer"></div>
      </div>
    </div>

    <div v-else-if="type === 'gallery'" class="skeleton-gallery grid-4">
      <div v-for="i in count" :key="i" class="skeleton-image-card skeleton-shimmer"></div>
    </div>

    <div v-else-if="type === 'stats'" class="skeleton-stats grid-4">
      <div v-for="i in 4" :key="i" class="skeleton-stat-card skeleton-shimmer">
        <div class="skeleton-stat-icon"></div>
        <div class="skeleton-stat-value skeleton-shimmer"></div>
      </div>
    </div>

    <div v-else-if="type === 'list'" class="skeleton-list">
      <div v-for="i in count" :key="i" class="skeleton-list-item">
        <div class="skeleton-avatar skeleton-shimmer"></div>
        <div class="skeleton-item-content">
          <div class="skeleton-item-title skeleton-shimmer"></div>
          <div class="skeleton-item-meta skeleton-shimmer"></div>
        </div>
      </div>
    </div>

    <div v-else-if="type === 'text'" class="skeleton-text">
      <div class="skeleton-text-line skeleton-shimmer"></div>
      <div class="skeleton-text-line skeleton-shimmer short"></div>
      <div class="skeleton-text-line skeleton-shimmer medium"></div>
    </div>

    <div v-else class="skeleton-default">
      <div class="skeleton-block skeleton-shimmer"></div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  type: {
    type: String,
    default: "default", // card, gallery, stats, list, text, default
    validator: (value) => {
      const validTypes = ["card", "gallery", "stats", "list", "text", "default"];
      return validTypes.includes(value);
    },
  },
  count: {
    type: Number,
    default: 4,
  },
});
</script>

<style scoped>
.skeleton-container {
  padding: 20px 0;
}

/* Common skeleton styles */
.skeleton-shimmer {
  background: linear-gradient(
    90deg,
    #f0f0f0 0%,
    #e8e8e8 50%,
    #f0f0f0 100%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
}

@keyframes shimmer {
  0% {
    background-position: -200% 0;
  }
  100% {
    background-position: 200% 0;
  }
}

/* Card skeleton */
.skeleton-card {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.skeleton-header {
  margin-bottom: 20px;
}

.skeleton-title {
  height: 24px;
  width: 60%;
  margin-bottom: 10px;
}

.skeleton-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.skeleton-line {
  height: 16px;
  width: 100%;
}

/* Gallery skeleton */
.skeleton-gallery {
  gap: 20px;
}

.skeleton-image-card {
  aspect-ratio: 1;
  border-radius: 12px;
}

/* Stats skeleton */
.skeleton-stats {
  gap: 20px;
}

.skeleton-stat-card {
  background: white;
  border-radius: 12px;
  padding: 30px 20px;
  text-align: center;
}

.skeleton-stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  margin: 0 auto 15px;
}

.skeleton-stat-value {
  height: 48px;
  width: 80%;
  margin: 0 auto;
}

/* List skeleton */
.skeleton-list-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  background: white;
  border-radius: 8px;
  margin-bottom: 12px;
}

.skeleton-avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  flex-shrink: 0;
}

.skeleton-item-content {
  flex: 1;
}

.skeleton-item-title {
  height: 18px;
  width: 60%;
  margin-bottom: 8px;
}

.skeleton-item-meta {
  height: 14px;
  width: 40%;
}

/* Text skeleton */
.skeleton-text {
  background: white;
  border-radius: 8px;
  padding: 20px;
}

.skeleton-text-line {
  height: 20px;
  margin-bottom: 12px;
  width: 100%;
}

.skeleton-text-line.short {
  width: 70%;
}

.skeleton-text-line.medium {
  width: 85%;
}

/* Default skeleton */
.skeleton-default {
  background: white;
  border-radius: 8px;
  padding: 16px;
}

.skeleton-block {
  height: 100px;
  border-radius: 4px;
}

/* Responsive Design */
@media (max-width: 768px) {
  .skeleton-gallery,
  .skeleton-stats {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 480px) {
  .skeleton-gallery,
  .skeleton-stats {
    grid-template-columns: 1fr;
  }

  .skeleton-stat-card {
    padding: 20px;
  }
}
</style>