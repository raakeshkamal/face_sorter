<template>
  <div class="sorting-view">
    <div class="view-header">
      <h1>🔀 Sorting</h1>
      <p class="subtitle">Cluster and classify unknown faces</p>
    </div>

    <!-- Sorting Configuration -->
    <div v-if="!operationStarted" class="sorting-config card">
      <h2 class="config-title">Sorting Configuration</h2>
      <form @submit.prevent="startSorting">
        <div class="form-grid grid-2">
          <div class="form-group">
            <label class="form-label">Cache Directory</label>
            <input
              v-model="form.cache_dir"
              type="text"
              class="form-input"
              placeholder="/path/to/cache"
              required
            />
            <p class="form-help">Directory containing cached face embeddings</p>
          </div>

          <div class="form-group">
            <label class="form-label">Max Results</label>
            <input
              v-model="form.max_results"
              type="number"
              class="form-input"
              min="1"
              max="100"
              placeholder="10"
            />
            <p class="form-help">Number of top clusters to display (1-100)</p>
          </div>
        </div>

        <div class="info-box">
          <p class="info-text">
            <strong>Sorting</strong> will cluster unknown faces using HDBSCAN and display
            the top clusters for manual classification.
          </p>
          <ul class="info-features">
            <li>✅ Automatic face clustering</li>
            <li>✅ Display top clusters by similarity</li>
            <li>✅ Quick class assignment from clusters</li>
            <li>✅ Real-time progress tracking</li>
          </ul>
        </div>

        <div class="form-actions">
          <button type="submit" class="btn btn-primary btn-large">
            <span class="btn-icon">🔀</span>
            <span>Start Sorting</span>
          </button>
        </div>
      </form>
    </div>

    <!-- Progress Display -->
    <div v-else-if="operationStarted" class="progress-section">
      <ProgressBar
        operation-type="Sorting"
        :task-id="taskId"
        @cancel="handleCancel"
        @reset="handleReset"
      />
    </div>

    <!-- Results Section (shown after sorting completes) -->
    <div v-if="showResults" class="results-section">
      <h2 class="results-title">Sorting Results</h2>
      <div class="results-stats grid-3">
        <div class="stat-card">
          <div class="stat-icon">👥</div>
          <div class="stat-value">{{ results.total_clusters }}</div>
          <div class="stat-label">Clusters Found</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">📷</div>
          <div class="stat-value">{{ results.total_faces }}</div>
          <div class="stat-label">Faces Sorted</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">👥</div>
          <div class="stat-value">{{ results.assigned_classes }}</div>
          <div class="stat-label">Assigned to Classes</div>
        </div>
      </div>

      <!-- Cluster Gallery -->
      <div class="clusters-section">
        <h3 class="section-title">Top Clusters</h3>
        <div class="clusters-grid grid-4">
          <div
            v-for="cluster in results.clusters"
            :key="cluster.id"
            class="cluster-card card"
          >
            <div class="cluster-header">
              <span class="cluster-id">Cluster #{{ cluster.id }}</span>
              <span class="cluster-size">{{ cluster.size }} faces</span>
            </div>
            <div class="cluster-preview">
              <div class="cluster-images">
                <img
                  v-for="(face, index) in cluster.faces.slice(0, 4)"
                  :key="index"
                  :src="getImageUrl(face)"
                  class="cluster-image"
                  loading="lazy"
                />
              </div>
            </div>
            <div class="cluster-actions">
              <button class="btn btn-secondary" @click="viewCluster(cluster)">
                View Cluster
              </button>
              <button class="btn btn-primary" @click="assignToClass(cluster)">
                Assign to Class
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from "vue";
import ProgressBar from "../components/ProgressBar.vue";
import { apiService } from "../services/api";

const form = ref({
  cache_dir: "",
  max_results: 10,
});

const operationStarted = ref(false);
const taskId = ref("");
const loading = ref(false);
const showResults = ref(false);

// Mock results for now (would come from API)
const results = ref({
  total_clusters: 0,
  total_faces: 0,
  assigned_classes: 0,
  clusters: [],
});

const startSorting = async () => {
  try {
    loading.value = true;
    // TODO: Implement sorting API call
    // const response = await apiService.startSorting(form.value);
    // taskId.value = response.task_id;
    // operationStarted.value = true;

    // For now, show results directly
    showMockResults();
  } catch (error) {
    console.error("Failed to start sorting:", error);
    alert("Failed to start sorting. Please check your configuration and try again.");
  } finally {
    loading.value = false;
  }
};

const showMockResults = () => {
  operationStarted.value = true;
  showResults.value = true;

  // Mock results for demo
  results.value = {
    total_clusters: 8,
    total_faces: 156,
    assigned_classes: 12,
    clusters: [
      {
        id: 1,
        size: 24,
        faces: [],
      },
      {
        id: 2,
        size: 18,
        faces: [],
      },
      {
        id: 3,
        size: 15,
        faces: [],
      },
      // More clusters would be here
    ],
  };
};

const getImageUrl = (face) => {
  return `/images/${face.filename || face.path}`;
};

const viewCluster = (cluster) => {
  console.log("View cluster:", cluster);
  // TODO: Implement cluster viewing modal
};

const assignToClass = (cluster) => {
  console.log("Assign cluster to class:", cluster);
  // TODO: Implement class assignment flow
};

const handleCancel = () => {
  console.log("Sorting cancelled");
  operationStarted.value = false;
  showResults.value = false;
  taskId.value = "";
};

const handleReset = () => {
  operationStarted.value = false;
  showResults.value = false;
  taskId.value = "";
};
</script>

<style scoped>
.sorting-view {
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

/* Configuration Form */
.sorting-config {
  max-width: 900px;
  margin-bottom: 30px;
}

.config-title {
  font-size: 24px;
  font-weight: 600;
  color: var(--primary-color);
  margin: 0 0 24px 0;
  padding-bottom: 16px;
  border-bottom: 2px solid var(--primary-color);
}

.form-grid {
  margin-bottom: 24px;
}

.form-group {
  margin-bottom: 20px;
}

.form-label {
  display: block;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-color);
  margin-bottom: 8px;
}

.form-input {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 16px;
  font-family: inherit;
  transition: all 0.3s ease;
}

.form-input:focus {
  outline: none;
  border-color: var(--primary-color);
  box-shadow: 0 0 0 4px rgba(101, 123, 236, 0.1);
}

.form-help {
  font-size: 13px;
  color: #666;
  margin: 4px 0 0 0;
  line-height: 1.4;
}

.info-box {
  background: #f8f9fa;
  border-left: 4px solid var(--primary-color);
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 24px;
}

.info-text {
  font-size: 16px;
  line-height: 1.6;
  margin: 0 0 12px 0;
}

.info-features {
  list-style: none;
  padding: 0;
  margin: 0;
}

.info-features li {
  padding: 6px 0;
  font-size: 15px;
  color: #333;
}

.form-actions {
  display: flex;
  gap: 16px;
  margin-top: 32px;
}

.btn-large {
  padding: 16px 32px;
  font-size: 18px;
}

.btn-icon {
  margin-right: 8px;
  font-size: 20px;
}

/* Progress Section */
.progress-section {
  margin-bottom: 30px;
}

/* Results Section */
.results-section {
  animation: slideUp 0.5s ease;
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.results-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--secondary-color);
  margin: 0 0 30px 0;
  text-align: center;
}

.results-stats {
  margin-bottom: 40px;
}

.stat-card {
  text-align: center;
  padding: 24px;
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
}

.stat-icon {
  font-size: 40px;
  margin-bottom: 12px;
  display: block;
}

.stat-value {
  font-size: 36px;
  font-weight: 700;
  color: var(--primary-color);
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  font-weight: 600;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* Cluster Section */
.clusters-section {
  margin-top: 40px;
}

.section-title {
  font-size: 22px;
  font-weight: 600;
  color: var(--secondary-color);
  margin: 0 0 24px 0;
}

.clusters-grid {
  margin-bottom: 30px;
}

.cluster-card {
  padding: 20px;
  transition: all 0.3s ease;
}

.cluster-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.12);
}

.cluster-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #e0e0e0;
}

.cluster-id {
  font-size: 16px;
  font-weight: 600;
  color: var(--primary-color);
}

.cluster-size {
  font-size: 14px;
  color: #666;
  background: #f0f0f0;
  padding: 4px 12px;
  border-radius: 12px;
}

.cluster-preview {
  margin-bottom: 16px;
}

.cluster-images {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}

.cluster-image {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: 6px;
  background: #f0f0f0;
}

.cluster-actions {
  display: flex;
  gap: 8px;
}

.cluster-actions .btn {
  flex: 1;
  padding: 10px 16px;
  font-size: 14px;
}

/* Responsive Design */
@media (max-width: 768px) {
  .form-grid,
  .results-stats,
  .clusters-grid {
    grid-template-columns: 1fr;
  }

  .form-actions {
    flex-direction: column;
  }

  .btn-large {
    width: 100%;
  }

  .cluster-images {
    grid-template-columns: repeat(4, 1fr);
  }
}
</style>