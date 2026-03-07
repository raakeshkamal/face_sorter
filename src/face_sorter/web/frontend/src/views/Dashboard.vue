<template>
  <div class="dashboard">
    <div class="dashboard-header">
      <h1>Dashboard</h1>
      <p class="subtitle">Face Sorter Overview</p>
    </div>

    <div v-if="loading" class="skeleton-wrapper">
      <SkeletonLoader type="stats" />
    </div>

    <div v-else-if="!loading" class="stats-grid grid-4">
      <!-- Statistics Cards -->
      <div class="card stat-card">
        <div class="stat-header">
          <span class="stat-icon">👥</span>
          <span class="stat-title">Total Faces</span>
        </div>
        <div class="stat-value">{{ stats.total_faces }}</div>
      </div>

      <div class="card stat-card">
        <div class="stat-header">
          <span class="stat-icon">👥</span>
          <span class="stat-title">Total Classes</span>
        </div>
        <div class="stat-value">{{ stats.total_classes }}</div>
      </div>

      <div class="card stat-card">
        <div class="stat-header">
          <span class="stat-icon">🔀</span>
          <span class="stat-title">Total Clusters</span>
        </div>
        <div class="stat-value">{{ stats.total_clusters }}</div>
      </div>

      <div class="card stat-card">
        <div class="stat-header">
          <span class="stat-icon">📷</span>
          <span class="stat-title">Total Images</span>
        </div>
        <div class="stat-value">{{ stats.total_images }}</div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="quick-actions">
      <h2 class="section-title">Quick Actions</h2>
      <div class="actions-grid grid-2">
        <router-link to="/training" class="card action-card">
          <span class="action-icon">🎯</span>
          <span class="action-title">Start Training</span>
          <p class="action-desc">Detect faces and generate embeddings</p>
        </router-link>

        <router-link to="/cleaning" class="card action-card">
          <span class="action-icon">🧹</span>
          <span class="action-title">Clean Dataset</span>
          <p class="action-desc">Validate and standardize images</p>
        </router-link>

        <router-link to="/sorting" class="card action-card">
          <span class="action-icon">🔀</span>
          <span class="action-title">Sort Faces</span>
          <p class="action-desc">Cluster and classify unknown faces</p>
        </router-link>

        <router-link to="/classes" class="card action-card">
          <span class="action-icon">👥</span>
          <span class="action-title">Manage Classes</span>
          <p class="action-desc">View and manage face classes</p>
        </router-link>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { apiService } from "../services/api";
import SkeletonLoader from "../components/SkeletonLoader.vue";

const stats = ref({
  total_faces: 0,
  total_classes: 0,
  total_clusters: 0,
  total_images: 0,
});

const loading = ref(true);

const loadStats = async () => {
  try {
    loading.value = true;
    const data = await apiService.getOverview();
    stats.value = data;
  } catch (error) {
    console.error("Failed to load statistics:", error);
    // Show error toast
    if (typeof toast !== "undefined") {
      toast.error("Failed to load statistics. Please try again.");
    }
  } finally {
    loading.value = false;
  }
};

onMounted(() => {
  loadStats();
});
</script>

<style scoped>
.dashboard {
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

.dashboard-header {
  margin-bottom: 30px;
}

.dashboard-header h1 {
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

.stats-grid {
  margin-bottom: 40px;
}

.stat-card {
  text-align: center;
}

.stat-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin-bottom: 15px;
}

.stat-icon {
  font-size: 32px;
}

.stat-title {
  font-size: 14px;
  font-weight: 600;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.stat-value {
  font-size: 48px;
  font-weight: 700;
  color: var(--primary-color);
  line-height: 1;
}

.section-title {
  font-size: 24px;
  font-weight: 600;
  color: var(--secondary-color);
  margin: 0 0 20px 0;
}

.quick-actions {
  margin-top: 40px;
}

.actions-grid {
  margin-bottom: 30px;
}

.action-card {
  text-decoration: none;
  color: inherit;
  display: block;
  padding: 30px;
  text-align: center;
  transition: all 0.3s ease;
}

.action-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
}

.action-icon {
  font-size: 40px;
  margin-bottom: 15px;
  display: block;
}

.action-title {
  display: block;
  font-size: 20px;
  font-weight: 600;
  color: var(--primary-color);
  margin-bottom: 8px;
}

.action-desc {
  display: block;
  font-size: 14px;
  color: #666;
  margin: 0;
  line-height: 1.4;
}
</style>