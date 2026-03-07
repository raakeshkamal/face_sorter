<template>
  <div class="toast-container">
    <transition-group name="toast">
      <div
        v-for="toast in toasts"
        :key="toast.id"
        :class="['toast-item', `toast-${toast.type}`]"
        @click="handleClick(toast)"
      >
        <div class="toast-icon">
          {{ getIcon(toast.type) }}
        </div>
        <div class="toast-content">
          <p class="toast-message">{{ toast.message }}</p>
        </div>
        <button class="toast-close" @click.stop="handleClose(toast)">
          ✕
        </button>
      </div>
    </transition-group>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from "vue";
import toastService from "../services/toast";

const toasts = ref([]);

const getIcon = (type) => {
  const icons = {
    success: "✓",
    error: "✕",
    warning: "⚠",
    info: "ℹ",
  };
  return icons[type] || icons.info;
};

const handleClick = (toast) => {
  // Can add click handling if needed
  console.log("Toast clicked:", toast);
};

const handleClose = (toast) => {
  toastService.remove(toast.id);
};

// Subscribe to toast service
onMounted(() => {
  const unsubscribe = toastService.subscribe((newToasts) => {
    toasts.value = newToasts;
  });

  // Cleanup on unmount
  onUnmounted(() => {
    unsubscribe();
  });
});
</script>

<style scoped>
.toast-container {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  gap: 10px;
  pointer-events: none;
}

.toast-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  background: white;
  padding: 16px 20px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  cursor: pointer;
  pointer-events: auto;
  transition: all 0.3s ease;
  max-width: 400px;
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(50px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.toast-item:hover {
  transform: translateX(-4px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}

/* Toast Types */
.toast-success {
  border-left: 4px solid var(--success-color);
}

.toast-success .toast-icon {
  color: var(--success-color);
}

.toast-error {
  border-left: 4px solid var(--error-color);
}

.toast-error .toast-icon {
  color: var(--error-color);
}

.toast-warning {
  border-left: 4px solid var(--warning-color);
}

.toast-warning .toast-icon {
  color: var(--warning-color);
}

.toast-info {
  border-left: 4px solid var(--primary-color);
}

.toast-info .toast-icon {
  color: var(--primary-color);
}

.toast-icon {
  font-size: 24px;
  line-height: 1;
  flex-shrink: 0;
}

.toast-content {
  flex: 1;
}

.toast-message {
  font-size: 15px;
  color: #333;
  margin: 0;
  line-height: 1.4;
}

.toast-close {
  background: transparent;
  border: none;
  color: #666;
  cursor: pointer;
  font-size: 18px;
  padding: 4px;
  margin-left: 8px;
  transition: all 0.2s ease;
  flex-shrink: 0;
}

.toast-close:hover {
  color: var(--error-color);
}

/* Transition Styles */
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from {
  opacity: 0;
  transform: translateX(50px);
}

.toast-leave-to {
  opacity: 0;
  transform: translateX(100px);
}

/* Responsive Design */
@media (max-width: 768px) {
  .toast-container {
    right: 10px;
    left: 10px;
  }

  .toast-item {
    max-width: 100%;
    padding: 12px 16px;
  }
}
</style>