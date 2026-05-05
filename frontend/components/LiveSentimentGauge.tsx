"use client";

import { motion } from "framer-motion";

interface LiveSentimentGaugeProps {
  value: number; // -1 to +1
  eventCount: number;
}

function getGaugeColor(val: number): string {
  if (val > 0.3) return "#34d399";
  if (val > 0.1) return "#6ee7b7";
  if (val > -0.1) return "#fbbf24";
  if (val > -0.3) return "#fb923c";
  return "#fb7185";
}

function getLabel(val: number): string {
  if (val > 0.3) return "Very Positive";
  if (val > 0.1) return "Positive";
  if (val > -0.1) return "Neutral";
  if (val > -0.3) return "Negative";
  return "Very Negative";
}

export default function LiveSentimentGauge({
  value,
  eventCount,
}: LiveSentimentGaugeProps) {
  // Normalize value from [-1, 1] to [0, 100] for the bar position
  const position = Math.max(0, Math.min(100, ((value + 1) / 2) * 100));
  const color = getGaugeColor(value);

  return (
    <div className="glass rounded-2xl p-5 border border-border/50 space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-bold text-foreground">Live Sentiment</h4>
        <span className="text-[10px] text-muted font-mono">
          Last {Math.min(eventCount, 20)} events
        </span>
      </div>

      {/* Gauge bar */}
      <div className="relative">
        {/* Background gradient bar */}
        <div className="h-3 rounded-full overflow-hidden bg-secondary/50">
          <div
            className="h-full rounded-full"
            style={{
              background:
                "linear-gradient(90deg, #fb7185 0%, #fb923c 25%, #fbbf24 50%, #6ee7b7 75%, #34d399 100%)",
            }}
          />
        </div>

        {/* Marker needle */}
        <motion.div
          className="absolute top-1/2 -translate-y-1/2"
          animate={{ left: `${position}%` }}
          transition={{ type: "spring", stiffness: 200, damping: 25 }}
          style={{ marginLeft: "-8px" }}
        >
          <div
            className="h-6 w-4 rounded-sm border-2 shadow-lg"
            style={{
              backgroundColor: color,
              borderColor: "rgba(255,255,255,0.3)",
              boxShadow: `0 0 12px ${color}60`,
            }}
          />
        </motion.div>

        {/* Labels */}
        <div className="flex justify-between mt-2">
          <span className="text-[9px] text-rose-400 font-medium">-1.0</span>
          <span className="text-[9px] text-muted">0.0</span>
          <span className="text-[9px] text-emerald-400 font-medium">+1.0</span>
        </div>
      </div>

      {/* Value display */}
      <div className="flex items-center justify-between">
        <motion.span
          key={value.toFixed(3)}
          initial={{ scale: 1.2, opacity: 0.7 }}
          animate={{ scale: 1, opacity: 1 }}
          className="text-2xl font-bold font-mono"
          style={{ color }}
        >
          {value >= 0 ? "+" : ""}
          {value.toFixed(3)}
        </motion.span>
        <span
          className="text-xs font-medium px-2 py-0.5 rounded-md"
          style={{ color, backgroundColor: `${color}15` }}
        >
          {getLabel(value)}
        </span>
      </div>
    </div>
  );
}
