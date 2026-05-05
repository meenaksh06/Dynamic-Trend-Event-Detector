"use client";

import { useEffect, useState } from "react";
import { BrainCircuit, Info, AlertTriangle, CheckCircle, BarChart3 } from "lucide-react";
import HybridForecastChart from "@/components/HybridForecastChart";
import TopicList from "@/components/TopicList";
import { fetchTopics, fetchHybridForecast } from "@/lib/api";
import { useLiveContext } from "@/components/WebSocketProvider";
import { motion } from "framer-motion";

export default function ForecastPage() {
  const [topics, setTopics] = useState<any[]>([]);
  const [hybridData, setHybridData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [loadingForecast, setLoadingForecast] = useState(false);
  const [selectedTopicId, setSelectedTopicId] = useState<number>(0);
  const { topicCounts, connectionState } = useLiveContext();

  useEffect(() => {
    async function loadInitialData() {
      try {
        const topicsData = await fetchTopics();
        setTopics(topicsData);
        if (topicsData.length > 0) {
          loadForecast(topicsData[0].id);
        }
      } catch (error) {
        console.error("Initial load failed:", error);
      } finally {
        setLoading(false);
      }
    }
    loadInitialData();
  }, []);

  async function loadForecast(topicId: number) {
    setLoadingForecast(true);
    try {
      const data = await fetchHybridForecast(topicId);
      setHybridData(data);
      setSelectedTopicId(topicId);
    } catch (error) {
      console.error("Hybrid forecast load failed:", error);
    } finally {
      setLoadingForecast(false);
    }
  }

  const selectedTopic = topics.find(t => t.id === selectedTopicId);
  const metrics = hybridData && !hybridData.error ? hybridData : null;
  const liveCount = (selectedTopicId !== null && topicCounts[selectedTopicId]) ? topicCounts[selectedTopicId] : 0;

  return (
    <div className="space-y-8 pb-12">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="h-12 w-12 rounded-2xl bg-violet-500/20 flex items-center justify-center text-violet-500">
            <BarChart3 className="h-7 w-7" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-foreground">Predictive Analytics</h1>
            <p className="text-muted text-sm">Phase 3 — Hybrid ARIMA + BiLSTM forecasting engine.</p>
          </div>
        </div>
        {connectionState === "connected" && (
          <div className="glass px-4 py-1.5 rounded-full border border-emerald-500/30 flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] text-emerald-400 font-bold tracking-widest uppercase">Live Trajectory Active</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Topic Selection */}
        <div className="space-y-6">
          <div className="glass rounded-3xl p-6 border border-border/50">
            <h3 className="text-lg font-bold mb-6">Select Pipeline</h3>
            <TopicList
              topics={topics}
              isLoading={loading}
              selectedId={selectedTopicId}
              onSelect={loadForecast}
            />
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="rounded-2xl bg-amber-500/5 border border-amber-500/20 p-4 flex gap-3"
          >
            <AlertTriangle className="h-5 w-5 text-amber-500 shrink-0" />
            <p className="text-xs text-amber-200/70 leading-relaxed">
              Hybrid predictions combine ARIMA linear trends with BiLSTM non-linear corrections.
              High-volatility events may cause variance in mid-term accuracy.
            </p>
          </motion.div>
        </div>

        {/* Forecast Display */}
        <div className="lg:col-span-2 space-y-6">
          <HybridForecastChart
            data={hybridData}
            isLoading={loading || loadingForecast}
            liveCount={liveCount}
          />

          {/* Model Metrics Cards */}
          {metrics && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* ARIMA Card */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass rounded-2xl p-5 border border-amber-500/20"
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className="h-3 w-3 rounded-full bg-amber-500" />
                  <h4 className="text-sm font-bold text-foreground">ARIMA (ML)</h4>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted">RMSE</span>
                    <span className="font-mono font-bold">{metrics.arima.metrics.RMSE}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted">R²</span>
                    <span className="font-mono font-bold">{metrics.arima.metrics.R2}</span>
                  </div>
                </div>
              </motion.div>

              {/* BiLSTM Card */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="glass rounded-2xl p-5 border border-violet-500/20"
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className="h-3 w-3 rounded-full bg-violet-500" />
                  <h4 className="text-sm font-bold text-foreground">BiLSTM (DL)</h4>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted">RMSE</span>
                    <span className="font-mono font-bold">{metrics.bilstm.metrics.RMSE}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted">R²</span>
                    <span className="font-mono font-bold">{metrics.bilstm.metrics.R2}</span>
                  </div>
                </div>
              </motion.div>

              {/* Hybrid Card */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="glass rounded-2xl p-5 border border-cyan-500/30 bg-primary/5"
              >
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle className="h-4 w-4 text-cyan-400" />
                  <h4 className="text-sm font-bold text-primary">Hybrid</h4>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted">RMSE</span>
                    <span className="font-mono font-bold text-emerald-400">{metrics.hybrid.metrics.RMSE}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted">R²</span>
                    <span className="font-mono font-bold text-emerald-400">{metrics.hybrid.metrics.R2}</span>
                  </div>
                </div>
              </motion.div>
            </div>
          )}

          {/* Technical Summary */}
          <div className="glass rounded-2xl p-6 border border-border/50">
            <h4 className="font-bold text-foreground mb-2">Hybrid Architecture</h4>
            <p className="text-xs text-muted leading-relaxed">
              <strong className="text-foreground">Phase 3 Pipeline:</strong> ARIMA(2,1,2) extracts linear trend →
              Residuals computed → BiLSTM(64+32 bidirectional) learns non-linear corrections →
              Additive Fusion: <code className="text-primary">y_hybrid = y_ARIMA + ε_BiLSTM</code><br/>
              <strong className="text-foreground">Optimizer:</strong> Adam (lr=0.001) &nbsp;|&nbsp;
              <strong className="text-foreground">Regularization:</strong> Dropout(0.2) + EarlyStopping
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
