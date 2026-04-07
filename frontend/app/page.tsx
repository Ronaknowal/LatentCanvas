"use client";

import dynamic from "next/dynamic";
import { useCallback, useRef } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import AIOutputView from "./components/AIOutputView";
import ControlPanel from "./components/ControlPanel";
import ConnectionStatus from "./components/ConnectionStatus";

const Canvas = dynamic(() => import("./components/Canvas"), {
  ssr: false,
  loading: () => (
    <div className="h-full w-full flex items-center justify-center bg-gray-900">
      <p className="text-gray-500">Loading canvas...</p>
    </div>
  ),
});

export default function Home() {
  const { connectionState, sendBinary, sendConfig, latestImage } = useWebSocket();
  const lastSketchRef = useRef<ArrayBuffer | null>(null);

  const handleSketchExport = useCallback(
    (blob: Blob) => {
      blob.arrayBuffer().then((buf) => {
        lastSketchRef.current = buf;
        sendBinary(buf);
      });
    },
    [sendBinary]
  );

  // Re-send the last sketch after config changes so the new prompt/strength
  // takes effect immediately without requiring the user to draw again
  const resendLastSketch = useCallback(() => {
    if (lastSketchRef.current) {
      sendBinary(lastSketchRef.current);
    }
  }, [sendBinary]);

  const handlePromptChange = useCallback(
    (prompt: string) => {
      sendConfig({ prompt });
      // Small delay to let server process the config update before the sketch
      setTimeout(resendLastSketch, 50);
    },
    [sendConfig, resendLastSketch]
  );

  const handleStrengthChange = useCallback(
    (strength: number) => {
      sendConfig({ strength });
      setTimeout(resendLastSketch, 50);
    },
    [sendConfig, resendLastSketch]
  );

  return (
    <div className="h-screen flex flex-col">
      {/* Top bar */}
      <div className="flex items-center justify-between p-3 border-b border-gray-800">
        <h1 className="text-lg font-semibold">LatentCanvas</h1>
        <ConnectionStatus state={connectionState} />
      </div>

      {/* Controls */}
      <div className="p-3 border-b border-gray-800">
        <ControlPanel
          onPromptChange={handlePromptChange}
          onStrengthChange={handleStrengthChange}
        />
      </div>

      {/* Split view: Canvas | AI Output */}
      <div className="flex-1 flex min-h-0">
        <div className="w-1/2 border-r border-gray-800">
          <Canvas onSketchExport={handleSketchExport} />
        </div>
        <div className="w-1/2 p-4">
          <AIOutputView imageUrl={latestImage} />
        </div>
      </div>
    </div>
  );
}
