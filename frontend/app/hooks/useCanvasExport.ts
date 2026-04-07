"use client";

import { useCallback, useEffect, useRef } from "react";
import type { Editor } from "tldraw";
import { CANVAS_EXPORT_DEBOUNCE_MS } from "../lib/config";

export function useCanvasExport(
  editor: Editor | null,
  onExport: (blob: Blob) => void
) {
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onExportRef = useRef(onExport);
  onExportRef.current = onExport;

  const exportCanvas = useCallback(async () => {
    if (!editor) return;

    const shapeIds = editor.getCurrentPageShapeIds();
    if (shapeIds.size === 0) return;

    const result = await editor.toImage([...shapeIds], {
      format: "jpeg",
      scale: 1,
      background: true,
      padding: 0,
    });

    if (result?.blob) {
      onExportRef.current(result.blob);
    }
  }, [editor]);

  useEffect(() => {
    if (!editor) return;

    const cleanup = editor.store.listen(
      () => {
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
        }
        timeoutRef.current = setTimeout(exportCanvas, CANVAS_EXPORT_DEBOUNCE_MS);
      },
      { source: "user", scope: "document" }
    );

    return () => {
      cleanup();
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [editor, exportCanvas]);
}
