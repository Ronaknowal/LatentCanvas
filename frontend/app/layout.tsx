import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LatentCanvas",
  description: "Real-time generative canvas",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="h-full bg-gray-950 text-white">{children}</body>
    </html>
  );
}
