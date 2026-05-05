import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import { WebSocketProvider } from "@/components/WebSocketProvider";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Dynamic Trend & Event Detector",
  description: "Advanced deep learning platform for trend forecasting and event detection.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <WebSocketProvider>
          <div className="flex bg-background min-h-screen">
            <Sidebar />
            <main className="flex-1 ml-64 p-8 transition-all duration-300">
              <div className="max-w-7xl mx-auto">
                {children}
              </div>
            </main>
          </div>
        </WebSocketProvider>
      </body>
    </html>
  );
}
