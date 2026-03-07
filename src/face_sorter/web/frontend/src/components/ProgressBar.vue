<template>
  <div class="progress-bar-container">
    <div v-if="!started" class="progress-idle">
      <span class="idle-icon">⏳</span>
      <span class="idle-text">{{ idleText }}</span>
    </div>

    <div v-else-if="loading" class="progress-active">
      <div class="progress-header">
        <div class="progress-info">
          <span class="progress-title">{{ operationType }}</span>
          <span class="progress-status">{{ currentStatus }}</span>
        </div>
        <button v-if="cancellable" class="cancel-btn" @click="handleCancel">
          ✕
        </button>
      </div>

      <div class="progress-track">
        <div class="progress-fill" :style="{ width: `${progressPercentage}%` }">
          <div class="progress-glow"></div>
        </div>
      </div>

      <div class="progress-details">
        <div class="progress-stat">
          <span class="stat-label">Progress:</span>
          <span class="stat-value">{{ current }} / {{ total }}</span>
        </div>
        <div class="progress-stat">
          <span class="stat-label">Percentage:</span>
          <span class="stat-value">{{ progressPercentage }}%</span>
        </div>
        <div v-if="currentItem" class="progress-stat">
          <span class="stat-label">Current:</span>
          <span class="stat-value">{{ currentItem }}</span>
        </div>
        <div v-if="eta" class="progress-stat">
          <span class="stat-label">ETA:</span>
          <span class="stat-value">{{ eta }}</span>
        </div>
      </div>

      <!-- Progress Logs -->
      <div v-if="logs.length > 0" class="progress-logs">
        <div
          v-for="(log, index) in logs"
          :key="index"
          class="log-entry"
        >
          <span class="log-time">{{ log.time }}</span>
          <span class="log-message">{{ log.message }}</span>
        </div>
      </div>
    </div>

    <div v-else-if="completed" class="progress-complete">
      <div class="complete-icon">✓</div>
      <h3 class="complete-title">{{ operationType }} Complete</h3>
      <div class="complete-summary">
        <div class="summary-item">
          <span class="summary-label">Processed:</span>
          <span class="summary-value">{{ current }} / {{ total }}</span>
        </div>
        <div v-if="result.successful !== undefined" class="summary-item">
          <span class="summary-label">Successful:</span>
          <span class="summary-value">{{ result.successful }}</span>
        </div>
        <div v-if="result.failed !== undefined" class="summary-item">
          <span class="summary-label">Failed:</span>
          <span class="summary-value">{{ result.failed }}</span>
        </div>
      </div>
      <button class="btn btn-primary" @click="handleReset">
        Start New Operation
      </button>
    </div>

    <div v-else-if="error" class="progress-error">
      <div class="error-icon">⚠</div>
      <h3 class="error-title">{{ operationType }} Failed</h3>
      <p class="error-message">{{ errorMessage }}</p>
      <div v-if="errorDetails" class="error-details">
        <strong>Details:</strong>
        <p>{{ errorDetails }}</p>
      </div>
      <button class="btn btn-primary" @click="handleReset">
        Try Again
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onUnmounted } from "vue";
import websocketService from "../services/websocket";

const props = defineProps({
  operationType: {
    type: String,
    default: "Operation",
  },
  taskId: {
    type: String,
    required: true,
  },
  cancellable: {
    type: Boolean,
    default: false,
  },
  idleText: {
    type: String,
    default: "Ready to start",
  },
});

const emit = defineEmits(["cancel", "reset"]);

const started = ref(false);
const loading = ref(false);
const completed = ref(false);
const error = ref(false);
const current = ref(0);
const total = ref(0);
const currentStatus = ref("");
const currentItem = ref("");
const logs = ref([]);
const result = ref({});
const errorMessage = ref("");
const errorDetails = ref("");

const progressPercentage = computed(() => {
  if (total.value === 0) return 0;
  return Math.round((current.value / total.value) * 100);
});

const eta = computed(() => {
  if (!loading.value || current.value === 0) return "";
  // Simple ETA calculation (can be enhanced)
  const remaining = total.value - current.value;
  if (remaining <= 0) return "Complete";
  return `${remaining} remaining`;
});

const handleMessage = (data) => {
  console.log("Progress update:", data);

  switch (data.type) {
    case "progress":
      started.value = true;
      loading.value = true;
      completed.value = false;
      error.value = false;
      current.value = data.progress.current;
      total.value = data.progress.total;
      currentStatus.value = data.progress.status;
      currentItem.value = data.progress.current_item || "";

      // Add log entry
      logs.value.unshift({
        time: new Date().toLocaleTimeString(),
        message: `${currentStatus.value}: ${currentItem.value}`,
      });

      // Keep only last 20 logs
      if (logs.value.length > 20) {
        logs.value = logs.value.slice(0, 20);
      }
      break;

    case "complete":
      started.value = true;
      loading.value = false;
      completed.value = true;
      error.value = false;
      result.value = data.result;
      currentStatus.value = "Complete";
      break;

    case "error":
      started.value = true;
      loading.value = false;
      completed.value = false;
      error.value = true;
      errorMessage.value = data.error.message || "Unknown error";
      errorDetails.value = data.error.details || "";
      break;
  }
};

const handleError = (wsError) => {
  console.error("WebSocket error:", wsError);
  started.value = true;
  loading.value = false;
  completed.value = false;
  error.value = true;
  errorMessage.value = "Connection error";
  errorDetails.value = wsError.message || "Failed to connect to server";
};

const handleCancel = () => {
  emit("cancel");
};

const handleReset = () => {
  emit("reset");
};

// Connect WebSocket when component mounts
onMounted(() => {
  websocketService.connect(props.operationType.toLowerCase(), props.taskId, handleMessage, handleError);
});

// Disconnect WebSocket when component unmounts
onUnmounted(() => {
  websocketService.disconnect();
});
</script>

<style scoped>
.progress-bar-container {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Idle State */
.progress-idle {
  text-align: center;
  padding: 40px 20px;
  color: #666;
}

.idle-icon {
  font-size: 48px;
  display: block;
  margin-bottom: 15px;
}

.idle-text {
  font-size: 18px;
  font-weight: 500;
}

/* Active Progress State */
.progress-active {
  animation: fadeIn 0.3s ease;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.progress-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.progress-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--secondary-color);
}

.progress-status {
  padding: 4px 12px;
  background: var(--primary-color);
  color: white;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.cancel-btn {
  background: white;
  border: 2px solid var(--error-color);
  color: var(--error-color);
  border-radius: 50%;
  width: 36px;
  height: 36px;
  font-size: 18px;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.cancel-btn:hover {
  background: var(--error-color);
  color: white;
}

/* Progress Track */
.progress-track {
  width: 100%;
  height: 12px;
  background: #e0e0e0;
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 20px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
  border-radius: 6px;
  transition: width 0.5s ease;
  position: relative;
  overflow: hidden;
}

.progress-glow {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
  filter: blur(8px);
  opacity: 0.4;
}

/* Progress Details */
.progress-details {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}

.progress-stat {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stat-label {
  font-size: 12px;
  font-weight: 600;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.stat-value {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-color);
}

/* Progress Logs */
.progress-logs {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 16px;
  max-height: 200px;
  overflow-y: auto;
  margin-top: 20px;
}

.log-entry {
  display: flex;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid #e0e0e0;
  font-size: 14px;
}

.log-entry:last-child {
  border-bottom: none;
}

.log-time {
  color: #666;
  font-size: 12px;
  min-width: 80px;
}

.log-message {
  color: #333;
  flex: 1;
}

/* Complete State */
.progress-complete {
  text-align: center;
  padding: 40px 20px;
  animation: fadeIn 0.3s ease;
}

.complete-icon {
  font-size: 64px;
  color: var(--success-color);
  margin-bottom: 20px;
  display: block;
}

.complete-title {
  font-size: 24px;
  font-weight: 700;
  color: var(--success-color);
  margin: 0 0 20px 0;
}

.complete-summary {
  display: flex;
  justify-content: center;
  gap: 30px;
  margin-bottom: 30px;
  flex-wrap: wrap;
}

.summary-item {
  text-align: center;
}

.summary-label {
  font-size: 14px;
  color: #666;
  display: block;
  margin-bottom: 4px;
}

.summary-value {
  font-size: 20px;
  font-weight: 600;
  color: var(--primary-color);
  display: block;
}

/* Error State */
.progress-error {
  text-align: center;
  padding: 40px 20px;
  animation: fadeIn 0.3s ease;
}

.error-icon {
  font-size: 64px;
  color: var(--error-color);
  margin-bottom: 20px;
  display: block;
}

.error-title {
  font-size: 24px;
  font-weight: 700;
  color: var(--error-color);
  margin: 0 0 15px 0;
}

.error-message {
  font-size: 16px;
  color: #333;
  margin: 0 0 20px 0;
  line-height: 1.6;
}

.error-details {
  background: #fff3cd;
  border-left: 4px solid var(--error-color);
  padding: 16px;
  margin: 20px 0;
  text-align: left;
  border-radius: 4px;
}

.error-details p {
  margin: 8px 0 0 0;
  color: #721c24;
  font-size: 14px;
}

/* Responsive Design */
@media (max-width: 768px) {
  .progress-details {
    grid-template-columns: 1fr;
  }

  .complete-summary {
    flex-direction: column;
    gap: 16px;
  }
}
</style>