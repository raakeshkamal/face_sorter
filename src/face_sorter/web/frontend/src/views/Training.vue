<template>
  <div class="training-view">
    <div class="view-header">
      <h1>🎯 Training</h1>
      <p class="subtitle">Detect faces and generate embeddings</p>
    </div>

    <!-- Training Form -->
    <div v-if="!operationStarted" class="training-form card">
      <h2 class="form-title">Training Configuration</h2>
      <form @submit.prevent="startTraining">
        <div class="form-grid grid-2">
          <div class="form-group">
            <label class="form-label">Source Directory</label>
            <input
              v-model="form.source_dir"
              type="text"
              class="form-input"
              placeholder="/path/to/images"
              required
            />
            <p class="form-help">Directory containing images to process</p>
          </div>

          <div class="form-group">
            <label class="form-label">No-Face Directory</label>
            <input
              v-model="form.noface_dir"
              type="text"
              class="form-input"
              placeholder="/path/to/noface"
            />
            <p class="form-help">Directory for images without faces</p>
          </div>

          <div class="form-group">
            <label class="form-label">Broken Images Directory</label>
            <input
              v-model="form.broken_dir"
              type="text"
              class="form-input"
              placeholder="/path/to/broken"
            />
            <p class="form-help">Directory for corrupted images</p>
          </div>

          <div class="form-group">
            <label class="form-label">Cache Directory</label>
            <input
              v-model="form.cache_dir"
              type="text"
              class="form-input"
              placeholder="/path/to/cache"
            />
            <p class="form-help">Directory for cached images</p>
          </div>
        </div>

        <div class="form-group">
          <label class="form-label">Duplicates Directory</label>
          <input
            v-model="form.duplicates_dir"
            type="text"
            class="form-input"
            placeholder="/path/to/duplicates"
          />
          <p class="form-help">Directory for duplicate images (will be skipped)</p>
        </div>

        <div class="form-actions">
          <button type="submit" class="btn btn-primary btn-large">
            <span class="btn-icon">🎯</span>
            <span>Start Training</span>
          </button>
        </div>
      </form>
    </div>

    <!-- Progress Display -->
    <div v-else class="progress-container">
      <ProgressBar
        operation-type="Training"
        :task-id="taskId"
        @cancel="handleCancel"
        @reset="handleReset"
      />
    </div>

    <!-- Info Section -->
    <div class="info-section card">
      <h3 class="info-title">About Training</h3>
      <div class="info-content">
        <p class="info-paragraph">
          <strong>Training</strong> performs face detection on your images using
          InsightFace and generates 512-dimensional face embeddings that can be used
          for recognition and clustering.
        </p>
        <ul class="info-list">
          <li>✅ Detects faces in images</li>
          <li>✅ Generates 512-dim embeddings</li>
          <li>✅ Extracts face metadata (age, gender, landmarks)</li>
          <li>✅ Moves images without faces to noface directory</li>
          <li>✅ Real-time progress tracking</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import ProgressBar from "../components/ProgressBar.vue";
import { apiService } from "../services/api";

const form = ref({
  source_dir: "",
  noface_dir: "",
  broken_dir: "",
  cache_dir: "",
  duplicates_dir: "",
});

const operationStarted = ref(false);
const taskId = ref("");
const loading = ref(false);

const startTraining = async () => {
  try {
    loading.value = true;
    const response = await apiService.startTraining(form.value);
    taskId.value = response.task_id;
    operationStarted.value = true;
  } catch (error) {
    console.error("Failed to start training:", error);
    alert("Failed to start training. Please check your configuration and try again.");
  } finally {
    loading.value = false;
  }
};

const handleCancel = () => {
  console.log("Training cancelled");
  operationStarted.value = false;
  taskId.value = "";
};

const handleReset = () => {
  operationStarted.value = false;
  taskId.value = "";
};
</script>

<style scoped>
.training-view {
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

/* Form Styling */
.training-form {
  max-width: 900px;
  margin-bottom: 30px;
}

.form-title {
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
  margin-bottom: 24px;
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

/* Progress Container */
.progress-container {
  margin-bottom: 30px;
}

/* Info Section */
.info-section {
  max-width: 800px;
}

.info-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--secondary-color);
  margin: 0 0 20px 0;
}

.info-content {
  line-height: 1.8;
}

.info-paragraph {
  margin-bottom: 20px;
}

.info-list {
  list-style: none;
  padding: 0;
}

.info-list li {
  padding: 8px 0;
  font-size: 16px;
  color: #333;
}

/* Responsive Design */
@media (max-width: 768px) {
  .form-grid {
    grid-template-columns: 1fr;
  }

  .form-actions {
    flex-direction: column;
  }

  .btn-large {
    width: 100%;
  }
}
</style>