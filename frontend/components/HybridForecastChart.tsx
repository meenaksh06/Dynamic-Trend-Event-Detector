"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";
import { motion } from "framer-motion";

interface HybridForecastChartProps {
  data: any;
  isLoading?: boolean;
  liveCount?: number;
}

export default function HybridForecastChart({ data, isLoading, liveCount = 0 }: HybridForecastChartProps) {
  if (isLoading || !data || data.error) {
    return (
      <div className="glass rounded-3xl p-8 h-[450px] relative overflow-hidden">
        <div className="shimmer absolute inset-0" />
      </div>
    );
  }

  const { train_dates, test_dates, actual, train_values, arima, bilstm, hybrid } = data;

  // Build chart data: train region + test region
  const chartData: any[] = [];

  // Training data points
  train_values.forEach((val: number, i: number) => {
    chartData.push({
      date: train_dates[i],
      actual: val,
      region: "train",
    });
  });

  // Test data points with all model predictions
  test_dates.forEach((date: string, i: number) => {
    chartData.push({
      date,
      actual: actual[i],
      arima: arima.predictions[i],
      bilstm: bilstm.predictions[i],
      hybrid: hybrid.predictions[i],
      region: "test",
    });
  });

  if (liveCount > 0 && chartData.length > 0) {
    const lastActual = chartData[chartData.length - 1].actual || 0;
    chartData.push({
      date: "LIVE NOW",
      actual: lastActual + (liveCount * 2), // Magnify slightly for visibility
      arima: null,
      bilstm: null,
      hybrid: null,
      region: "test",
    });
  }

  // Future data points (Phase 3 True Prediction)
  if (data.future_dates) {
    data.future_dates.forEach((date: string, i: number) => {
      chartData.push({
        date,
        actual: null, // We don't know the future yet!
        arima: data.future_arima[i],
        bilstm: data.future_bilstm[i],
        hybrid: data.future_hybrid[i],
        region: "future",
      });
    });
  }

  const splitDate = test_dates[0];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-3xl p-6 border border-border/50"
    >
      <div className="mb-6">
        <h3 className="text-xl font-bold text-foreground">
          Hybrid Forecast: ARIMA + BiLSTM
        </h3>
        <p className="text-sm text-muted">
          Phase 3 — Neuro-Statistical Decomposition with additive fusion
        </p>
      </div>

      <div className="h-[380px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
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
              interval={Math.floor(chartData.length / 10)}
            />
            <YAxis stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(15, 23, 42, 0.95)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "12px",
                backdropFilter: "blur(10px)",
                fontSize: "12px",
              }}
              labelFormatter={(label) => `Period: ${label}`}
            />
            <Legend verticalAlign="top" height={36} iconType="line" />
            <ReferenceLine
              x={splitDate}
              stroke="#f43f5e"
              strokeDasharray="6 4"
              label={{ value: "Train / Test", fill: "#f43f5e", fontSize: 10, position: "top" }}
            />
            {data.future_dates && data.future_dates.length > 0 && (
              <ReferenceLine
                x={data.future_dates[0]}
                stroke="#10b981"
                strokeDasharray="3 3"
                label={{ value: "FUTURE", fill: "#10b981", fontSize: 10, position: "top" }}
              />
            )}

            {/* Actual data */}
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual"
              stroke="#ffffff"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 5, fill: "#fff" }}
            />
            {/* ARIMA */}
            <Line
              type="monotone"
              dataKey="arima"
              name="ARIMA (ML)"
              stroke="#f59e0b"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={false}
            />
            {/* BiLSTM */}
            <Line
              type="monotone"
              dataKey="bilstm"
              name="BiLSTM (DL)"
              stroke="#8b5cf6"
              strokeWidth={2}
              strokeDasharray="4 4"
              dot={false}
            />
            {/* Hybrid */}
            <Line
              type="monotone"
              dataKey="hybrid"
              name="Hybrid"
              stroke="#22d3ee"
              strokeWidth={3}
              dot={false}
              activeDot={{ r: 5, fill: "#22d3ee" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
