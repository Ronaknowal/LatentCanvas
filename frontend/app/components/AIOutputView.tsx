"use client";

interface AIOutputViewProps {
  imageUrl: string | null;
}

export default function AIOutputView({ imageUrl }: AIOutputViewProps) {
  return (
    <div className="h-full w-full flex items-center justify-center bg-gray-900 rounded-lg overflow-hidden">
      {imageUrl ? (
        <img
          src={imageUrl}
          alt="AI generated"
          className="max-h-full max-w-full object-contain"
        />
      ) : (
        <div className="text-gray-500 text-center p-8">
          <p className="text-lg">AI Output</p>
          <p className="text-sm mt-2">Draw something to see the AI generate an image</p>
        </div>
      )}
    </div>
  );
}
