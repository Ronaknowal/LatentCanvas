"use client";

import type { ConnectionState } from "../hooks/useWebSocket";

const STATUS_CONFIG: Record<ConnectionState, { label: string; color: string }> = {
  waiting: { label: "Waiting for server", color: "bg-yellow-500" },
  connecting: { label: "Connecting", color: "bg-yellow-500" },
  connected: { label: "Connected", color: "bg-green-500" },
  reconnecting: { label: "Reconnecting", color: "bg-red-500" },
};

export default function ConnectionStatus({ state }: { state: ConnectionState }) {
  const { label, color } = STATUS_CONFIG[state];

  return (
    <div className="flex items-center gap-2 text-sm">
      <div className={`h-2 w-2 rounded-full ${color}`} />
      <span className="text-gray-300">{label}</span>
    </div>
  );
}
