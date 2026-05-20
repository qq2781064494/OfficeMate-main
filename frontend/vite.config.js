import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    host: "127.0.0.1",
    proxy: {
      "/health": "http://127.0.0.1:8001",
      "/admin": "http://127.0.0.1:8001",
      "/documents": "http://127.0.0.1:8001",
      "/chat": "http://127.0.0.1:8001",
      "/feedback": "http://127.0.0.1:8001",
      "/sessions": "http://127.0.0.1:8001",
      "/agent": "http://127.0.0.1:8001",
      "/benchmark": "http://127.0.0.1:8001",
      "/local-eval": "http://127.0.0.1:8001",
      "/tasks": "http://127.0.0.1:8001"
    }
  },
  build: {
    outDir: "dist",
    emptyOutDir: true
  }
});
