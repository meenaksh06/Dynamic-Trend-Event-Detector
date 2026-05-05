"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { motion } from "framer-motion";

interface SentimentChartProps {
  data: any[];
  isLoading?: boolean;
}

export default function SentimentChart({ data, isLoading }: SentimentChartProps) {
  if (isLoading || !data || data.length === 0) {
    return (
      <div className="glass rounded-3xl p-8 h-[300px] relative overflow-hidden">
        <div className="shimmer absolute inset-0" />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-3xl p-6 border border-border/50"
    >
      <div className="mb-6">
        <h3 className="text-lg font-bold text-foreground">Sentiment Analysis</h3>
        <p className="text-sm text-muted">
          Monthly VADER compound sentiment across all articles
        </p>
      </div>

      <div className="h-[250px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#34d399" stopOpacity={0.3} />
                <stop offset="50%" stopColor="#fbbf24" stopOpacity={0.05} />
                <stop offset="95%" stopColor="#fb7185" stopOpacity={0.3} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis
              dataKey="date"
              stroke="#64748b"
              fontSize={9}
              tickLine={false}
              axisLine={false}
              tickFormatter={(str) => {
                const parts = str.split("-");
                const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
                return parts.length > 1 ? `${months[parseInt(parts[1]) - 1]} ${parts[0].slice(2)}` : str;
              }}
              interval={Math.floor(data.length / 8)}
            />
            <YAxis
              stroke="#64748b"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              domain={['auto', 'auto']}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(15, 23, 42, 0.95)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "12px",
                backdropFilter: "blur(10px)",
                fontSize: "12px",
              }}
              formatter={(value) => [Number(value).toFixed(4), "Avg Sentiment"]}
              labelFormatter={(label) => `Period: ${label}`}
            />
            <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
            <Area
              type="monotone"
              dataKey="avg_sentiment"
              name="Sentiment"
              stroke="#34d399"
              fill="url(#sentimentGradient)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
