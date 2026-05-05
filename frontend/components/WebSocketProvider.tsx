"use client";

import React, { createContext, useContext, useEffect, useRef, useState, useCallback, ReactNode } from "react";

export interface LiveEvent {
  timestamp: string;
  headline: string;
  category: string;
  sentiment_compound: number;
  topic_id: number;
  topic_name: string;
  link: string;
  event_type: "breaking" | "trending" | "update";
  word_count: number;
}

interface WebSocketContextType {
  lastEvent: LiveEvent | null;
  events: LiveEvent[];
  isConnected: boolean;
  connectionState: "connecting" | "connected" | "disconnected" | "reconnecting";
  topicCounts: Record<number, number>;
  avgSentiment: number;
  totalReceived: number;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

const MAX_EVENTS = 100;
const WS_URL = "ws://localhost:8000/ws/live";

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [lastEvent, setLastEvent] = useState<LiveEvent | null>(null);
  const [connectionState, setConnectionState] = useState<
    "connecting" | "connected" | "disconnected" | "reconnecting"
  >("disconnected");
  const [topicCounts, setTopicCounts] = useState<Record<number, number>>({});
  const [avgSentiment, setAvgSentiment] = useState(0);
  const [totalReceived, setTotalReceived] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const maxRetries = 20;
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setConnectionState(retriesRef.current > 0 ? "reconnecting" : "connecting");

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      retriesRef.current = 0;
      setConnectionState("connected");
      console.log("🟢 WebSocket connected");
    };

    ws.onmessage = (event) => {
      if (!mountedRef.current) return;
      try {
        const data: LiveEvent = JSON.parse(event.data);
        setLastEvent(data);
        setTotalReceived((prev) => prev + 1);

        setEvents((prev) => {
          const updated = [data, ...prev].slice(0, MAX_EVENTS);
          return updated;
        });

        // Update topic counts
        setTopicCounts((prev) => ({
          ...prev,
          [data.topic_id]: (prev[data.topic_id] || 0) + 1,
        }));

        // Update rolling average sentiment (last 20 events)
        setEvents((prev) => {
          const recent = prev.slice(0, 20);
          const sum = recent.reduce((a, e) => a + e.sentiment_compound, 0);
          setAvgSentiment(recent.length > 0 ? sum / recent.length : 0);
          return prev;
        });
      } catch (err) {
        console.error("WS parse error:", err);
      }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setConnectionState("disconnected");
      console.log("🔴 WebSocket disconnected");

      // Auto-reconnect with exponential backoff
      if (retriesRef.current < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, retriesRef.current), 30000);
        retriesRef.current += 1;
        setTimeout(() => {
          if (mountedRef.current) connect();
        }, delay);
      }
    };

    ws.onerror = () => {
      // onclose will fire after onerror
    };
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  const value = {
    lastEvent,
    events,
    isConnected: connectionState === "connected",
    connectionState,
    topicCounts,
    avgSentiment,
    totalReceived,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useLiveContext() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error("useLiveContext must be used within a WebSocketProvider");
  }
  return context;
}
