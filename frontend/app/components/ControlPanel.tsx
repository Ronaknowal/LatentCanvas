"use client";

import { useCallback, useRef, useState } from "react";
import { DEFAULT_PROMPT, DEFAULT_STRENGTH } from "../lib/config";

interface ControlPanelProps {
  onPromptChange: (prompt: string) => void;
  onStrengthChange: (strength: number) => void;
}

export default function ControlPanel({
  onPromptChange,
  onStrengthChange,
}: ControlPanelProps) {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [strength, setStrength] = useState(DEFAULT_STRENGTH);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handlePromptChange = useCallback(
    (value: string) => {
      setPrompt(value);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => onPromptChange(value), 500);
    },
    [onPromptChange]
  );

  const handleStrengthChange = useCallback(
    (value: number) => {
      setStrength(value);
      onStrengthChange(value);
    },
    [onStrengthChange]
  );

  return (
    <div className="flex items-center gap-4 p-3 bg-gray-800 rounded-lg">
      <input
        type="text"
        value={prompt}
        onChange={(e) => handlePromptChange(e.target.value)}
        placeholder="Describe the style..."
        className="flex-1 bg-gray-700 text-white px-3 py-2 rounded text-sm
                   placeholder-gray-400 outline-none focus:ring-1 focus:ring-blue-500"
      />

      <div className="flex items-center gap-2 shrink-0">
        <label className="text-sm text-gray-400">AI Strength</label>
        <input
          type="range"
          min={0.1}
          max={1.0}
          step={0.05}
          value={strength}
          onChange={(e) => handleStrengthChange(parseFloat(e.target.value))}
          className="w-28"
        />
        <span className="text-sm text-gray-300 w-8">{strength.toFixed(2)}</span>
      </div>
    </div>
  );
}
