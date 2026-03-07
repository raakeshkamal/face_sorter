// Toast notification service for user feedback
class ToastService {
  constructor() {
    this.toasts = [];
    this.listeners = [];
  }

  // Add a new toast notification
  add(message, type = "info", duration = 3000) {
    const id = Date.now();
    const toast = {
      id,
      message,
      type,
      duration,
    };

    this.toasts.push(toast);
    this.notifyListeners();

    // Auto-remove after duration
    setTimeout(() => {
      this.remove(id);
    }, duration);
  }

  // Remove a specific toast
  remove(id) {
    const index = this.toasts.findIndex((t) => t.id === id);
    if (index !== -1) {
      this.toasts.splice(index, 1);
      this.notifyListeners();
    }
  }

  // Remove all toasts
  clear() {
    this.toasts = [];
    this.notifyListeners();
  }

  // Get current toasts
  getToasts() {
    return this.toasts;
  }

  // Subscribe to toast changes
  subscribe(listener) {
    this.listeners.push(listener);
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index !== -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  // Notify all listeners
  notifyListeners() {
    this.listeners.forEach((listener) => listener(this.toasts));
  }

  // Convenience methods for different toast types
  success(message, duration) {
    this.add(message, "success", duration);
  }

  error(message, duration) {
    this.add(message, "error", duration || 5000);
  }

  warning(message, duration) {
    this.add(message, "warning", duration);
  }

  info(message, duration) {
    this.add(message, "info", duration);
  }
}

export default new ToastService();