export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export const WS_URL = BACKEND_URL.replace(/^http/, "ws");

export const CANVAS_EXPORT_DEBOUNCE_MS = 150;
export const CANVAS_EXPORT_QUALITY = 0.7;
export const CANVAS_EXPORT_FORMAT: "jpeg" | "png" = "jpeg";

export const DEFAULT_PROMPT = "high quality, detailed, photorealistic";
export const DEFAULT_STRENGTH = 0.5;

export const WS_RECONNECT_BASE_MS = 1000;
export const WS_RECONNECT_MAX_MS = 30000;
export const HEALTH_POLL_INTERVAL_MS = 3000;
