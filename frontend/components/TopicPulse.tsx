"use client";

import { motion } from "framer-motion";

interface TopicPulseProps {
  topicCounts: Record<number, number>;
  lastTopicId?: number;
}

const COLORS = [
  "#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#fb7185",
  "#a78bfa", "#2dd4bf", "#f472b6", "#60a5fa", "#4ade80",
];

export default function TopicPulse({ topicCounts, lastTopicId }: TopicPulseProps) {
  const maxCount = Math.max(1, ...Object.values(topicCounts));
  const topics = Array.from({ length: 10 }, (_, i) => i);

  return (
    <div className="glass rounded-2xl p-5 border border-border/50 space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-bold text-foreground">Topic Activity</h4>
        <span className="text-[10px] text-muted font-mono">
          {Object.values(topicCounts).reduce((a, b) => a + b, 0)} events
        </span>
      </div>

      <div className="space-y-2">
        {topics.map((topicId) => {
          const count = topicCounts[topicId] || 0;
          const width = maxCount > 0 ? (count / maxCount) * 100 : 0;
          const color = COLORS[topicId % COLORS.length];
          const isActive = topicId === lastTopicId;

          return (
            <div key={topicId} className="flex items-center gap-2">
              <span className="text-[10px] text-muted w-10 text-right font-mono shrink-0">
                T{topicId}
              </span>
              <div className="flex-1 h-4 rounded-full bg-secondary/30 overflow-hidden relative">
                <motion.div
                  className="h-full rounded-full relative"
                  style={{ backgroundColor: color }}
                  animate={{
                    width: `${Math.max(width, count > 0 ? 4 : 0)}%`,
                    opacity: isActive ? 1 : 0.7,
                  }}
                  transition={{
                    width: { type: "spring", stiffness: 200, damping: 25 },
                    opacity: { duration: 0.3 },
                  }}
                >
                  {isActive && (
                    <motion.div
                      className="absolute inset-0 rounded-full"
                      style={{ backgroundColor: color }}
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{
                        duration: 1,
                        repeat: 2,
                        ease: "easeInOut",
                      }}
                    />
                  )}
                </motion.div>
              </div>
              <span
                className="text-[10px] font-bold font-mono w-6 text-right"
                style={{ color }}
              >
                {count}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
