import axios from "axios";

// Create axios instance with base URL
const api = axios.create({
  baseURL: "http://127.0.0.1:8000/api",
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30000,
});

// API service methods
export const apiService = {
  // Statistics
  async getOverview() {
    const response = await api.get("/stats/overview");
    return response.data;
  },

  // Images
  async getImages(params = {}) {
    const response = await api.get("/images", { params });
    return response.data;
  },

  async getImage(id) {
    const response = await api.get(`/images/${id}`);
    return response.data;
  },

  async getUnsortedImages(params = {}) {
    const response = await api.get("/images/unsorted", { params });
    return response.data;
  },

  // Classes
  async getClasses() {
    const response = await api.get("/classes");
    return response.data;
  },

  async createClass(data) {
    const response = await api.post("/classes", data);
    return response.data;
  },

  async deleteClass(className) {
    const response = await api.delete(`/classes/${className}`);
    return response.data;
  },

  // Operations
  async startTraining(data) {
    const response = await api.post("/operations/train", data);
    return response.data;
  },

  async startCleaning(data) {
    const response = await api.post("/operations/clean", data);
    return response.data;
  },
};

export default api;