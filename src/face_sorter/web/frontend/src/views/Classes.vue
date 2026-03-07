<template>
  <div class="classes-view">
    <div class="view-header">
      <h1>👥 Classes</h1>
      <p class="subtitle">View and manage face classes</p>
    </div>

    <div v-if="loading" class="loading">
      Loading classes...
    </div>

    <div v-else-if="classes.length === 0" class="no-classes">
      <span class="no-classes-icon">👥</span>
      <p class="no-classes-text">No classes found</p>
      <p class="no-classes-hint">
        Create classes from sorted clusters to organize your faces.
      </p>
    </div>

    <div v-else class="classes-grid grid-3">
      <div v-for="classItem in classes" :key="classItem.class_name" class="card class-card">
        <div class="class-header">
          <span class="class-icon">👥</span>
          <h3 class="class-name">{{ classItem.class_name }}</h3>
        </div>
        <div class="class-actions">
          <button class="btn btn-secondary" @click="viewClass(classItem)">
            View
          </button>
          <button class="btn btn-secondary" @click="deleteClass(classItem)">
            Delete
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { apiService } from "../services/api";

const classes = ref([]);
const loading = ref(true);

const loadClasses = async () => {
  try {
    loading.value = true;
    classes.value = await apiService.getClasses();
  } catch (error) {
    console.error("Failed to load classes:", error);
  } finally {
    loading.value = false;
  }
};

const viewClass = (classItem) => {
  console.log("View class:", classItem.class_name);
  // TODO: Implement class viewing
};

const deleteClass = async (classItem) => {
  if (confirm(`Are you sure you want to delete class "${classItem.class_name}"?`)) {
    try {
      await apiService.deleteClass(classItem.class_name);
      await loadClasses();
    } catch (error) {
      console.error("Failed to delete class:", error);
      alert("Failed to delete class");
    }
  }
};

onMounted(() => {
  loadClasses();
});
</script>

<style scoped>
.classes-view {
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

.no-classes {
  text-align: center;
  padding: 60px 20px;
  color: #999;
}

.no-classes-icon {
  font-size: 64px;
  display: block;
  margin-bottom: 20px;
}

.no-classes-text {
  font-size: 20px;
  margin: 0 0 10px 0;
}

.no-classes-hint {
  font-size: 16px;
  margin: 0;
}

.class-card {
  text-align: center;
  padding: 30px 20px;
  transition: all 0.3s ease;
}

.class-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
}

.class-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 20px;
}

.class-icon {
  font-size: 32px;
}

.class-name {
  font-size: 18px;
  font-weight: 600;
  color: var(--primary-color);
  margin: 0;
}

.class-actions {
  display: flex;
  gap: 10px;
  justify-content: center;
}

.btn-secondary {
  padding: 8px 16px;
  font-size: 14px;
}
</style>