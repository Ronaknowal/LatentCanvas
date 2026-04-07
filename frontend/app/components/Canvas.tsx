"use client";

import { useState } from "react";
import { Tldraw, Editor } from "tldraw";
import "tldraw/tldraw.css";
import { useCanvasExport } from "../hooks/useCanvasExport";

interface CanvasProps {
  onSketchExport: (blob: Blob) => void;
}

export default function Canvas({ onSketchExport }: CanvasProps) {
  const [editor, setEditor] = useState<Editor | null>(null);

  useCanvasExport(editor, onSketchExport);

  return (
    <div className="h-full w-full relative">
      <Tldraw onMount={setEditor} />
    </div>
  );
}
