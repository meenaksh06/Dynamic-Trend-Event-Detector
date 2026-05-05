"use client";

import { useEffect, useState } from "react";
import { FlaskConical, Layers, Zap } from "lucide-react";
import AblationTable from "@/components/AblationTable";
import { fetchAblation } from "@/lib/api";
import { useLiveContext } from "@/components/WebSocketProvider";
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

export default function AblationPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const { connectionState } = useLiveContext();

  useEffect(() => {
    async function load() {
      try {
        const ablationData = await fetchAblation();
        setData(ablationData);
      } catch (error) {
        console.error("Ablation load failed:", error);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Prepare bar chart data from averages
  const barData = data?.averages?.map((row: any) => ({
    model: row.model,
    RMSE: row.RMSE,
    MAE: row.MAE,
  })) || [];

  return (
    <div className="space-y-8 pb-12">
      {/* Header */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="h-12 w-12 rounded-2xl bg-rose-500/20 flex items-center justify-center text-rose-500">
            <FlaskConical className="h-7 w-7" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-foreground">Ablation Studies</h1>
            <p className="text-muted text-sm">
              Diagnostic analysis proving the necessity of each component in the hybrid architecture.
            </p>
          </div>
        </div>
        
        {connectionState === "connected" && (
          <div className="glass px-4 py-1.5 rounded-full border border-emerald-500/30 flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] text-emerald-400 font-bold tracking-widest uppercase">Live Pipeline Monitoring</span>
          </div>
        )}
      </div>

      {/* Architecture explanation */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-4"
      >
        <div className="glass rounded-2xl p-5 border border-amber-500/20">
          <div className="flex items-center gap-2 mb-3">
            <div className="h-3 w-3 rounded-full bg-amber-500" />
            <h4 className="text-sm font-bold text-amber-400">ML Component</h4>
          </div>
          <p className="text-xs text-muted leading-relaxed">
            <strong className="text-foreground">ARIMA(2,1,2)</strong> — Captures linear trends,
            seasonal patterns, and autoregressive dependencies in topic time-series.
          </p>
        </div>
        <div className="glass rounded-2xl p-5 border border-violet-500/20">
          <div className="flex items-center gap-2 mb-3">
            <div className="h-3 w-3 rounded-full bg-violet-500" />
            <h4 className="text-sm font-bold text-violet-400">DL Component</h4>
          </div>
          <p className="text-xs text-muted leading-relaxed">
            <strong className="text-foreground">BiLSTM (64+32)</strong> — Learns non-linear
            residual patterns, event-driven spikes, and complex dynamics ARIMA misses.
          </p>
        </div>
        <div className="glass rounded-2xl p-5 border border-cyan-500/20">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="h-3 w-3 text-cyan-400" />
            <h4 className="text-sm font-bold text-cyan-400">Hybrid Fusion</h4>
          </div>
          <p className="text-xs text-muted leading-relaxed">
            <strong className="text-foreground">y = ARIMA + BiLSTM</strong> — Additive fusion
            combining linear baseline with non-linear correction for optimal predictions.
          </p>
        </div>
      </motion.div>

      {/* Bar Chart */}
      {barData.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass rounded-3xl p-6 border border-border/50"
        >
          <h3 className="text-lg font-bold mb-4">Model Comparison</h3>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="model" stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(15, 23, 42, 0.95)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "12px",
                    fontSize: "12px",
                  }}
                />
                <Legend verticalAlign="top" height={36} />
                <Bar dataKey="RMSE" name="RMSE" fill="#f43f5e" radius={[8, 8, 0, 0]} />
                <Bar dataKey="MAE" name="MAE" fill="#818cf8" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      )}

      {/* Ablation Table + Diagnostics */}
      <AblationTable data={data} isLoading={loading} />
    </div>
  );
}
