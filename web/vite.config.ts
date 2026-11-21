import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` serves the UI on :5173 and proxies /api to the Python server (python app.py, :8000).
// Build: `npm run build` writes web/dist, which app.py serves (no Node needed to run the project).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    assetsInlineLimit: 0, // fonts as files: the CSP allows fonts from 'self' only (no data: URIs)
  },
});
