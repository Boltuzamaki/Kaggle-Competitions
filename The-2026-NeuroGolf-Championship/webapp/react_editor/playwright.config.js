import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30000,
  use: {
    baseURL: process.env.NEUROGOLF_E2E_BASE_URL || "http://127.0.0.1:8081",
    viewport: { width: 1280, height: 900 },
  },
});
