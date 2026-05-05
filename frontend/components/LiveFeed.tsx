"use client";

import { motion, AnimatePresence } from "framer-motion";
import { ExternalLink, Zap, Flame, Newspaper } from "lucide-react";
import type { LiveEvent } from "@/components/WebSocketProvider";

interface LiveFeedProps {
  events: LiveEvent[];
}

const eventTypeConfig = {
  breaking: {
    label: "BREAKING",
    icon: Zap,
    color: "text-rose-400",
    bg: "bg-rose-500/10",
    border: "border-rose-500/30",
  },
  trending: {
    label: "TRENDING",
    icon: Flame,
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
  },
  update: {
    label: "UPDATE",
    icon: Newspaper,
    color: "text-sky-400",
    bg: "bg-sky-500/10",
    border: "border-sky-500/30",
  },
};

function getSentimentColor(val: number) {
  if (val > 0.2) return "bg-emerald-500";
  if (val < -0.2) return "bg-rose-500";
  return "bg-slate-400";
}

function timeAgo(ts: string) {
  const diff = Date.now() - new Date(ts).getTime();
  const secs = Math.floor(diff / 1000);
  if (secs < 5) return "just now";
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  return `${mins}m ago`;
}

export default function LiveFeed({ events }: LiveFeedProps) {
  if (events.length === 0) {
    return (
      <div className="glass rounded-3xl p-12 border border-border/50 text-center">
        <div className="animate-pulse">
          <Zap className="h-12 w-12 text-muted mx-auto mb-4 opacity-20" />
          <p className="text-muted text-sm">
            Waiting for live events...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3 max-h-[70vh] overflow-y-auto pr-2 custom-scrollbar">
      <AnimatePresence mode="popLayout" initial={false}>
        {events.slice(0, 50).map((event, index) => {
          const config = eventTypeConfig[event.event_type] || eventTypeConfig.update;
          const TypeIcon = config.icon;

          return (
            <motion.div
              key={event.timestamp + index}
              layout
              initial={{ opacity: 0, x: -40, scale: 0.95 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 40, scale: 0.9 }}
              transition={{ duration: 0.3, type: "spring", stiffness: 300, damping: 30 }}
              className={`glass group rounded-2xl p-4 border ${
                index === 0 ? config.border : "border-border/30"
              } hover:border-primary/30 transition-all duration-300`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0 space-y-2">
                  {/* Badges row */}
                  <div className="flex items-center gap-2 flex-wrap">
                    <span
                      className={`inline-flex items-center gap-1 text-[9px] uppercase tracking-widest font-bold px-2 py-0.5 rounded-md ${config.bg} ${config.color}`}
                    >
                      <TypeIcon className="h-2.5 w-2.5" />
                      {config.label}
                    </span>
                    <span className="text-[10px] uppercase tracking-widest font-bold text-primary px-2 py-0.5 rounded-md bg-primary/10">
                      {event.category}
                    </span>
                    <span className="text-[10px] text-muted italic">
                      {timeAgo(event.timestamp)}
                    </span>
                  </div>

                  {/* Headline */}
                  <h4 className="text-sm font-semibold text-foreground leading-snug group-hover:text-primary transition-colors line-clamp-2">
                    {event.headline}
                  </h4>

                  {/* Sentiment + Topic */}
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5">
                      <div
                        className={`h-1.5 w-1.5 rounded-full ${getSentimentColor(
                          event.sentiment_compound
                        )}`}
                      />
                      <span className="text-[10px] text-muted font-mono">
                        {event.sentiment_compound.toFixed(2)}
                      </span>
                    </div>
                    <span className="text-[10px] text-muted/70">
                      {event.topic_name.split(":")[0]}
                    </span>
                  </div>
                </div>

                {/* Link button */}
                <a
                  href={event.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-secondary/50 hover:bg-primary hover:text-primary-foreground transition-all duration-300 mt-1"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
