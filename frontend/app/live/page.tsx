"use client";

import { Radio, Wifi, WifiOff, Loader2, Activity } from "lucide-react";
import { motion } from "framer-motion";
import { useLiveContext } from "@/components/WebSocketProvider";
import LiveFeed from "@/components/LiveFeed";
import LiveSentimentGauge from "@/components/LiveSentimentGauge";
import TopicPulse from "@/components/TopicPulse";

const stateConfig = {
  connected: {
    icon: Wifi,
    label: "LIVE",
    color: "text-emerald-400",
    dotClass: "live-dot bg-emerald-500",
    bg: "bg-emerald-500/10",
  },
  connecting: {
    icon: Loader2,
    label: "CONNECTING",
    color: "text-amber-400",
    dotClass: "bg-amber-500 animate-pulse",
    bg: "bg-amber-500/10",
  },
  reconnecting: {
    icon: Loader2,
    label: "RECONNECTING",
    color: "text-amber-400",
    dotClass: "bg-amber-500 animate-pulse",
    bg: "bg-amber-500/10",
  },
  disconnected: {
    icon: WifiOff,
    label: "OFFLINE",
    color: "text-rose-400",
    dotClass: "bg-rose-500",
    bg: "bg-rose-500/10",
  },
};

export default function LivePage() {
  const {
    events,
    lastEvent,
    isConnected,
    connectionState,
    topicCounts,
    avgSentiment,
    totalReceived,
  } = useLiveContext();

  const config = stateConfig[connectionState];
  const StatusIcon = config.icon;

  return (
    <div className="space-y-8 pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="h-12 w-12 rounded-2xl bg-rose-500/20 flex items-center justify-center text-rose-500">
            <Radio className="h-7 w-7" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-foreground flex items-center gap-3">
              Live Intelligence Feed
              <span
                className={`inline-flex items-center gap-1.5 text-[10px] uppercase tracking-widest font-bold px-3 py-1 rounded-full ${config.bg} ${config.color}`}
              >
                <span className={`h-2 w-2 rounded-full ${config.dotClass}`} />
                {config.label}
              </span>
            </h1>
            <p className="text-muted text-sm">
              Real-time news event stream powered by WebSocket — simulated from
              the HuffPost dataset.
            </p>
          </div>
        </div>

        {/* Connection stats */}
        <div className="flex items-center gap-4">
          <div className="glass rounded-xl px-4 py-2 border border-border/50 flex items-center gap-2">
            <Activity className="h-4 w-4 text-primary" />
            <div>
              <p className="text-[10px] text-muted uppercase tracking-wider font-medium">
                Events
              </p>
              <motion.p
                key={totalReceived}
                initial={{ scale: 1.1 }}
                animate={{ scale: 1 }}
                className="text-lg font-bold font-mono text-foreground -mt-0.5"
              >
                {totalReceived}
              </motion.p>
            </div>
          </div>
          <div className="glass rounded-xl px-4 py-2 border border-border/50 flex items-center gap-2">
            <StatusIcon
              className={`h-4 w-4 ${config.color} ${
                connectionState === "connecting" ||
                connectionState === "reconnecting"
                  ? "animate-spin"
                  : ""
              }`}
            />
            <div>
              <p className="text-[10px] text-muted uppercase tracking-wider font-medium">
                Status
              </p>
              <p className={`text-xs font-bold ${config.color} -mt-0.5`}>
                {connectionState === "connected"
                  ? "Connected"
                  : connectionState === "reconnecting"
                  ? "Reconnecting..."
                  : connectionState === "connecting"
                  ? "Connecting..."
                  : "Disconnected"}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Live Feed — 2/3 */}
        <div className="lg:col-span-2">
          <LiveFeed events={events} />
        </div>

        {/* Right Panel — 1/3 */}
        <div className="space-y-6">
          <LiveSentimentGauge
            value={avgSentiment}
            eventCount={totalReceived}
          />

          <TopicPulse
            topicCounts={topicCounts}
            lastTopicId={lastEvent?.topic_id}
          />

          {/* Architecture Info */}
          <div className="glass rounded-2xl p-5 border border-border/50 bg-primary/5">
            <h4 className="text-sm font-bold text-primary uppercase tracking-widest mb-3">
              How It Works
            </h4>
            <p className="text-xs text-muted leading-relaxed">
              The backend samples articles from the{" "}
              <strong className="text-foreground">10,000 article</strong>{" "}
              HuffPost dataset every{" "}
              <strong className="text-foreground">3 seconds</strong> and
              broadcasts them via{" "}
              <strong className="text-foreground">WebSocket</strong>. Each event
              includes pre-computed{" "}
              <code className="text-primary">VADER sentiment</code> and{" "}
              <code className="text-primary">LDA topic assignment</code> from
              the Phase 1 pipeline.
            </p>
          </div>

          {/* Last event detail */}
          {lastEvent && (
            <motion.div
              key={lastEvent.timestamp}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass rounded-2xl p-5 border border-primary/20"
            >
              <h4 className="text-[10px] uppercase tracking-widest font-bold text-primary mb-2">
                Latest Event
              </h4>
              <div className="space-y-2">
                <div className="flex justify-between text-[11px]">
                  <span className="text-muted">Category</span>
                  <span className="font-bold text-foreground">
                    {lastEvent.category}
                  </span>
                </div>
                <div className="flex justify-between text-[11px]">
                  <span className="text-muted">Sentiment</span>
                  <span className="font-bold font-mono text-foreground">
                    {lastEvent.sentiment_compound.toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between text-[11px]">
                  <span className="text-muted">Topic</span>
                  <span className="font-bold text-foreground">
                    {lastEvent.topic_name.split(":")[0]}
                  </span>
                </div>
                <div className="flex justify-between text-[11px]">
                  <span className="text-muted">Words</span>
                  <span className="font-bold font-mono text-foreground">
                    {lastEvent.word_count}
                  </span>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
