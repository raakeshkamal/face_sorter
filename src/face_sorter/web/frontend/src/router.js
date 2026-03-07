import { createRouter, createWebHistory } from "vue-router";

// Import views
// import Dashboard from "./views/Dashboard.vue";
// import Training from "./views/Training.vue";
// import Cleaning from "./views/Cleaning.vue";
// import Sorting from "./views/Sorting.vue";
// import Classes from "./views/Classes.vue";
// import Unsorted from "./views/Unsorted.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/",
      name: "dashboard",
      component: () => import("./views/Dashboard.vue"),
    },
    {
      path: "/training",
      name: "training",
      component: () => import("./views/Training.vue"),
    },
    {
      path: "/cleaning",
      name: "cleaning",
      component: () => import("./views/Cleaning.vue"),
    },
    {
      path: "/sorting",
      name: "sorting",
      component: () => import("./views/Sorting.vue"),
    },
    {
      path: "/classes",
      name: "classes",
      component: () => import("./views/Classes.vue"),
    },
    {
      path: "/unsorted",
      name: "unsorted",
      component: () => import("./views/Unsorted.vue"),
    },
  ],
});

export default router;