<template>
  <div class="cleaning-view">
    <div class="view-header">
      <h1>🧹 Cleaning</h1>
      <p class="subtitle">Validate and standardize image dataset</p>
    </div>

    <!-- Cleaning Form -->
    <div v-if="!operationStarted" class="cleaning-form card">
      <h2 class="form-title">Cleaning Configuration</h2>
      <form @submit.prevent="startCleaning">
        <div class="form-grid grid-2">
          <div class="form-group">
            <label class="form-label">Source Directory</label>
            <input
              v-model="form.source_dir"
              type="text"
              class="form-input"
              placeholder="/path/to/raw/images"
              required
            />
            <p class="form-help">Directory containing images to clean</p>
          </div>

          <div class="form-group">
            <label class="form-label">Output Directory</label>
            <input
              v-model="form.output_dir"
              type="text"
              class="form-input"
              placeholder="/path/to/cleaned/images"
              required
            />
            <p class="form-help">Directory for cleaned images</p>
          </div>

          <div class="form-group">
            <label class="form-label">Broken Images Directory</label>
            <input
              v-model="form.broken_dir"
              type="text"
              class="form-input"
              placeholder="/path/to/broken/images"
            />
            <p class="form-help">Directory for invalid/corrupted images</p>
          </div>

          <div class="form-group">
            <label class="form-label">Batch Size</label>
            <input
              v-model="form.batch_size"
              type="number"
              class="form-input"
              min="1"
              max="100"
              placeholder="25"
            />
            <p class="form-help">Images to process at once (1-100)</p>
          </div>

          <div class="form-group">
            <label class="form-label">Image Prefix</label>
            <input
              v-model="form.img_prefix"
              type="text"
              class="form-input"
              placeholder="IMG"
            />
            <p class="form-help">Prefix for output filenames (e.g., IMG_001.jpg)</p>
          </div>

          <div class="form-group">
            <label class="form-label">JPEG Quality</label>
            <input
              v-model="form.quality"
              type="number"
              class="form-input"
              min="1"
              max="100"
              placeholder="95"
            />
            <p class="form-help">JPEG quality (1-100, higher = better quality)</p>
          </div>
        </div>

        <div class="form-group">
          <label class="form-label flex-row">
            <input
              v-model="form.recursive"
              type="checkbox"
              id="recursive"
              class="form-checkbox"
            />
            <span>Scan directories recursively</span>
          </label>
          <p class="form-help">Search subdirectories for images</p>
        </div>

        <div class="form-group">
          <label class="form-label">Starting Index</label>
          <input
            v-model="form.start_index"
            type="number"
            class="form-input"
            min="0"
            placeholder="1"
          />
          <p class="form-help">Starting number for sequential naming (0 for auto-detect)</p>
        </div>

        <div class="form-actions">
          <button type="submit" class="btn btn-primary btn-large">
            <span class="btn-icon">🧹</span>
            <span>Start Cleaning</span>
          </button>
        </div>
      </form>
    </div>

    <!-- Progress Display -->
    <div v-else class="progress-container">
      <ProgressBar
        operation-type="Cleaning"
        :task-id="taskId"
        @cancel="handleCancel"
        @reset="handleReset"
      />
    </div>

    <!-- Info Section -->
    <div class="info-section card">
      <h3 class="info-title">About Cleaning</h3>
      <div class="info-content">
        <p class="info-paragraph">
          <strong>Cleaning</strong> validates your image dataset, converts all images to RGB JPEG format,
          and saves them with sequential naming to a flat output directory.
        </p>
        <ul class="info-list">
          <li>✅ Validates image integrity</li>
          <li>✅ Converts to RGB JPEG format</li>
          <li>✅ Applies sequential naming (IMG_001.jpg, IMG_002.jpg, etc.)</li>
          <li>✅ Moves broken images to separate directory</li>
          <li>✅ Real-time progress tracking</li>
          <li>✅ Supports various image formats (JPG, PNG, BMP, etc.)</li>
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
  output_dir: "",
  broken_dir: "",
  batch_size: 25,
  img_prefix: "IMG",
  quality: 95,
  recursive: true,
  start_index: 0,
});

const operationStarted = ref(false);
const taskId = ref("");
const loading = ref(false);

const startCleaning = async () => {
  try {
    loading.value = true;
    const response = await apiService.startCleaning(form.value);
    taskId.value = response.task_id;
    operationStarted.value = true;
  } catch (error) {
    console.error("Failed to start cleaning:", error);
    alert("Failed to start cleaning. Please check your configuration and try again.");
  } finally {
    loading.value = false;
  }
};

const handleCancel = () => {
  console.log("Cleaning cancelled");
  operationStarted.value = false;
  taskId.value = "";
};

const handleReset = () => {
  operationStarted.value = false;
  taskId.value = "";
};
</script>

<style scoped>
.cleaning-view {
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
.cleaning-form {
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
  margin-bottom: 20px;
}

.form-label {
  display: block;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-color);
  margin-bottom: 8px;
}

.form-label.flex-row {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.form-checkbox {
  width: 18px;
  height: 18px;
  cursor: pointer;
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