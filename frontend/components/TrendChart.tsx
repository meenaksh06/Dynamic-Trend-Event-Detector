"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { motion } from "framer-motion";

interface TrendChartProps {
  data: any[];
  topics: any[];
  isLoading?: boolean;
}

const COLORS = [
  "#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#fb7185", 
  "#a78bfa", "#2dd4bf", "#f472b6", "#60a5fa", "#4ade80"
];

export default function TrendChart({ data, topics, isLoading }: TrendChartProps) {
  if (isLoading) {
    return (
      <div className="glass rounded-3xl p-8 h-[400px] relative overflow-hidden">
        <div className="shimmer absolute inset-0" />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass rounded-3xl p-6 border border-border/50"
    >
      <div className="flex items-center justify-between mb-8">
        <div>
          <h3 className="text-lg font-bold text-foreground">Topic Evolution</h3>
          <p className="text-sm text-muted">Temporal distribution of themes across dataset</p>
        </div>
      </div>

      <div className="h-[350px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              {COLORS.map((color, i) => (
                <linearGradient key={`color-${i}`} id={`colorTopic${i}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis 
              dataKey="date" 
              stroke="#64748b" 
              fontSize={10} 
              tickLine={false} 
              axisLine={false} 
              tickFormatter={(str) => {
                const [y, m] = str.split('-');
                const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
                return `${monthNames[parseInt(m)-1]} ${y.slice(2)}`;
              }}
            />
            <YAxis 
              stroke="#64748b" 
              fontSize={10} 
              tickLine={false} 
              axisLine={false} 
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: "rgba(15, 23, 42, 0.9)", 
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "12px",
                backdropFilter: "blur(10px)",
                fontSize: "12px"
              }}
              itemStyle={{ padding: "2px 0" }}
            />
            {topics.slice(0, 5).map((topic, i) => (
              <Area
                key={topic.id}
                type="monotone"
                dataKey={`topic_${topic.id}`}
                name={topic.name.split(':')[0]}
                stroke={COLORS[i % COLORS.length]}
                fill={`url(#colorTopic${i})`}
                strokeWidth={2}
                stackId="1"
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
