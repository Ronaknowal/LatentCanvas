"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  WS_URL,
  WS_RECONNECT_BASE_MS,
  WS_RECONNECT_MAX_MS,
  BACKEND_URL,
  HEALTH_POLL_INTERVAL_MS,
} from "../lib/config";

export type ConnectionState = "connecting" | "connected" | "reconnecting" | "waiting";

interface UseWebSocketReturn {
  connectionState: ConnectionState;
  sendBinary: (data: ArrayBuffer) => void;
  sendConfig: (config: { prompt?: string; strength?: number }) => void;
  latestImage: string | null;
}

export function useWebSocket(): UseWebSocketReturn {
  const [connectionState, setConnectionState] = useState<ConnectionState>("waiting");
  const [latestImage, setLatestImage] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevImageUrlRef = useRef<string | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setConnectionState("connecting");
    const ws = new WebSocket(`${WS_URL}/ws/generate`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setConnectionState("connected");
      reconnectAttemptRef.current = 0;
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const blob = new Blob([event.data], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        if (prevImageUrlRef.current) {
          URL.revokeObjectURL(prevImageUrlRef.current);
        }
        prevImageUrlRef.current = url;
        setLatestImage(url);
      }
    };

    ws.onclose = () => {
      setConnectionState("reconnecting");
      scheduleReconnect();
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, []);

  const scheduleReconnect = useCallback(() => {
    const attempt = reconnectAttemptRef.current;
    const delay = Math.min(
      WS_RECONNECT_BASE_MS * Math.pow(2, attempt),
      WS_RECONNECT_MAX_MS
    );
    reconnectAttemptRef.current = attempt + 1;

    reconnectTimeoutRef.current = setTimeout(() => {
      connect();
    }, delay);
  }, [connect]);

  useEffect(() => {
    let mounted = true;
    let pollId: ReturnType<typeof setInterval>;

    const checkHealth = async () => {
      try {
        const resp = await fetch(`${BACKEND_URL}/api/health`);
        const data = await resp.json();
        if (data.status === "ready" && mounted) {
          clearInterval(pollId);
          connect();
        }
      } catch {
        // Server not reachable yet
      }
    };

    setConnectionState("waiting");
    checkHealth();
    pollId = setInterval(checkHealth, HEALTH_POLL_INTERVAL_MS);

    return () => {
      mounted = false;
      clearInterval(pollId);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      wsRef.current?.close();
      if (prevImageUrlRef.current) {
        URL.revokeObjectURL(prevImageUrlRef.current);
      }
    };
  }, [connect]);

  const sendBinary = useCallback((data: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  const sendConfig = useCallback((config: { prompt?: string; strength?: number }) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(config));
    }
  }, []);

  return { connectionState, sendBinary, sendConfig, latestImage };
}
