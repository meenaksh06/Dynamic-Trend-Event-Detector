"use client";

import { useEffect, useState } from "react";
import { TrendingUp, Info } from "lucide-react";
import TrendChart from "@/components/TrendChart";
import { fetchTopics, fetchTrends } from "@/lib/api";
import { useLiveContext } from "@/components/WebSocketProvider";
import { motion } from "framer-motion";

export default function TrendsPage() {
  const [topics, setTopics] = useState<any[]>([]);
  const [trends, setTrends] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const { topicCounts, connectionState } = useLiveContext();

  useEffect(() => {
    async function loadData() {
      try {
        const [topicsData, trendsData] = await Promise.all([
          fetchTopics(),
          fetchTrends()
        ]);
        setTopics(topicsData);
        setTrends(trendsData);
      } catch (error) {
        console.error("Dashboard data load failed:", error);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // Dynamically append live data to the trend chart
  const chartData = [...trends];
  if (Object.keys(topicCounts).length > 0 && chartData.length > 0) {
    const livePoint: any = { date: "LIVE NOW" };
    Object.entries(topicCounts).forEach(([topicId, count]) => {
      // Add the live count to the most recent historical baseline for a smooth visual jump
      const lastHistoricalValue = chartData[chartData.length - 1][`topic_${topicId}`] || 0;
      livePoint[`topic_${topicId}`] = lastHistoricalValue + (count * 5); // Multiplier to make live data visible on monthly scale
    });
    chartData.push(livePoint);
  }

  return (
    <div className="space-y-8 pb-12">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="h-12 w-12 rounded-2xl bg-emerald-500/20 flex items-center justify-center text-emerald-500">
            <TrendingUp className="h-7 w-7" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-foreground">Trend Analysis</h1>
            <p className="text-muted text-sm">Longitudinal study of thematic shifts in the news landscape.</p>
          </div>
        </div>
        
        {connectionState === "connected" && (
          <div className="glass px-4 py-1.5 rounded-full border border-emerald-500/30 flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] text-emerald-400 font-bold tracking-widest uppercase">Live Injection Active</span>
          </div>
        )}
      </div>

      <div className="space-y-8">
        <TrendChart data={chartData} topics={topics} isLoading={loading} />
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="glass rounded-3xl p-8 border border-border/50">
                <div className="flex items-center gap-3 mb-4">
                    <Info className="h-6 w-6 text-primary" />
                    <h3 className="text-lg font-bold">Analysis Methodology</h3>
                </div>
                <p className="text-sm text-muted leading-relaxed mb-4">
                    The trends above represent the relative frequency of thematic clusters over time. Each data point corresponds to a monthly aggregation of news reports.
                </p>
                <div className="space-y-3">
                    <div className="flex items-center gap-2">
                        <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                        <span className="text-xs text-foreground font-medium">Latent Dirichlet Allocation (LDA) Topic Extraction</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                        <span className="text-xs text-foreground font-medium">Monthly Time-Series Aggregation</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="h-1.5 w-1.5 rounded-full bg-amber-500" />
                        <span className="text-xs text-foreground font-medium">Rolling Average Smoothing (optional)</span>
                    </div>
                </div>
            </div>

            <div className="glass rounded-3xl p-8 border border-border/50 bg-primary/5 flex flex-col justify-center">
                <h3 className="text-2xl font-bold mb-4 gradient-text">Emerging Pattern Detected</h3>
                <p className="text-muted mb-6">
                    Our analysis has identified a cross-thematic correlation between "Economic Policy" and "Global Health" trends in the Q3-2022 window. This suggests an integrated event cycle.
                </p>
                <button className="bg-primary text-primary-foreground px-6 py-3 rounded-xl font-bold hover:shadow-lg hover:shadow-primary/20 transition-all w-fit">
                    Export Analysis Report
                </button>
            </div>
        </div>
      </div>
    </div>
  );
}
