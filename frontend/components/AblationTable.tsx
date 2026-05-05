"use client";

import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle, XCircle } from "lucide-react";

interface AblationTableProps {
  data: any;
  isLoading?: boolean;
}

function MetricCell({ value, isBest, isWorst }: { value: number | string; isBest?: boolean; isWorst?: boolean }) {
  return (
    <td className={`px-4 py-3 text-sm font-mono text-right ${
      isBest ? "text-emerald-400 font-bold" :
      isWorst ? "text-rose-400" : "text-foreground"
    }`}>
      {typeof value === 'number' ? value.toFixed(4) : value}
    </td>
  );
}

export default function AblationTable({ data, isLoading }: AblationTableProps) {
  if (isLoading || !data) {
    return (
      <div className="glass rounded-3xl p-8 h-[300px] relative overflow-hidden">
        <div className="shimmer absolute inset-0" />
      </div>
    );
  }

  const { averages, diagnostics, per_topic } = data;

  // Find best/worst for each metric
  const findBest = (metric: string) => {
    if (!averages || averages.length === 0) return "";
    const sorted = [...averages].sort((a: any, b: any) =>
      metric === "R2" ? b[metric] - a[metric] : a[metric] - b[metric]
    );
    return sorted[0]?.model;
  };

  const bestModels: Record<string, string> = {
    RMSE: findBest("RMSE"),
    MAE: findBest("MAE"),
    R2: findBest("R2"),
    MAPE: findBest("MAPE"),
  };

  return (
    <div className="space-y-6">
      {/* Average Metrics Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-3xl p-6 border border-border/50 overflow-hidden"
      >
        <h3 className="text-lg font-bold mb-4">Average Metrics Across All Topics</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/30">
                <th className="px-4 py-3 text-left text-xs font-bold text-muted uppercase tracking-wider">Model</th>
                <th className="px-4 py-3 text-right text-xs font-bold text-muted uppercase tracking-wider">RMSE ↓</th>
                <th className="px-4 py-3 text-right text-xs font-bold text-muted uppercase tracking-wider">MAE ↓</th>
                <th className="px-4 py-3 text-right text-xs font-bold text-muted uppercase tracking-wider">R² ↑</th>
                <th className="px-4 py-3 text-right text-xs font-bold text-muted uppercase tracking-wider">MAPE ↓</th>
              </tr>
            </thead>
            <tbody>
              {averages?.map((row: any, i: number) => (
                <tr key={row.model} className={`border-b border-border/10 ${
                  row.model === "Hybrid" ? "bg-primary/5" : ""
                }`}>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      {row.model === "Hybrid" ? (
                        <CheckCircle className="h-4 w-4 text-emerald-400" />
                      ) : (
                        <div className={`h-3 w-3 rounded-full ${
                          row.model.includes("ARIMA") ? "bg-amber-500" : "bg-violet-500"
                        }`} />
                      )}
                      <span className={`text-sm font-semibold ${
                        row.model === "Hybrid" ? "text-primary" : "text-foreground"
                      }`}>{row.model}</span>
                    </div>
                  </td>
                  <MetricCell value={row.RMSE} isBest={bestModels.RMSE === row.model} />
                  <MetricCell value={row.MAE} isBest={bestModels.MAE === row.model} />
                  <MetricCell value={row.R2} isBest={bestModels.R2 === row.model} />
                  <MetricCell value={row.MAPE} isBest={bestModels.MAPE === row.model} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Diagnostics */}
      {diagnostics && diagnostics.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-4"
        >
          {diagnostics.map((diag: any, i: number) => (
            <div key={i} className="glass rounded-2xl p-5 border border-border/50">
              <div className="flex items-center gap-2 mb-3">
                <XCircle className="h-5 w-5 text-rose-500" />
                <h4 className="text-sm font-bold text-foreground">
                  Removing {diag.component_removed}
                </h4>
              </div>
              <p className="text-lg font-bold text-rose-400 mb-2">{diag.impact}</p>
              <p className="text-xs text-muted leading-relaxed">{diag.explanation}</p>
            </div>
          ))}
        </motion.div>
      )}

      {/* Per-Topic Breakdown */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass rounded-3xl p-6 border border-border/50 overflow-hidden"
      >
        <h3 className="text-lg font-bold mb-4">Per-Topic Breakdown</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/30">
                <th className="px-3 py-2 text-left text-xs font-bold text-muted uppercase">Topic</th>
                <th className="px-3 py-2 text-left text-xs font-bold text-muted uppercase">Model</th>
                <th className="px-3 py-2 text-right text-xs font-bold text-muted uppercase">RMSE</th>
                <th className="px-3 py-2 text-right text-xs font-bold text-muted uppercase">MAE</th>
                <th className="px-3 py-2 text-right text-xs font-bold text-muted uppercase">R²</th>
              </tr>
            </thead>
            <tbody>
              {per_topic?.map((row: any, i: number) => (
                <tr key={i} className={`border-b border-border/5 ${
                  row.model === "Hybrid" ? "bg-primary/5" : ""
                }`}>
                  <td className="px-3 py-2 text-xs text-muted">{row.topic}</td>
                  <td className="px-3 py-2 text-xs font-medium text-foreground">{row.model}</td>
                  <td className="px-3 py-2 text-xs font-mono text-right text-foreground">{row.RMSE?.toFixed(2)}</td>
                  <td className="px-3 py-2 text-xs font-mono text-right text-foreground">{row.MAE?.toFixed(2)}</td>
                  <td className="px-3 py-2 text-xs font-mono text-right text-foreground">{row.R2?.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  );
}
