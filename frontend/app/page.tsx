"use client";

import { useEffect, useState } from "react";
import {
  Users,
  TrendingUp,
  MessageSquare,
  Activity,
  BarChart as BarChartIcon,
  Search,
  Bell
} from "lucide-react";
import KPICard from "@/components/KPICard";
import TrendChart from "@/components/TrendChart";
import TopicList from "@/components/TopicList";
import SentimentChart from "@/components/SentimentChart";
import { fetchStats, fetchTopics, fetchTrends, fetchSentimentTimeline } from "@/lib/api";
import { useLiveContext } from "@/components/WebSocketProvider";
import { motion } from "framer-motion";

export default function Dashboard() {
  const [stats, setStats] = useState<any>(null);
  const [topics, setTopics] = useState<any[]>([]);
  const [trends, setTrends] = useState<any[]>([]);
  const [sentiment, setSentiment] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedTopicId, setSelectedTopicId] = useState<number>(0);
  const { totalReceived, avgSentiment: liveAvgSentiment, connectionState } = useLiveContext();

  useEffect(() => {
    async function loadData() {
      try {
        const [statsData, topicsData, trendsData, sentimentData] = await Promise.all([
          fetchStats(),
          fetchTopics(),
          fetchTrends(),
          fetchSentimentTimeline()
        ]);
        setStats(statsData);
        setTopics(topicsData);
        setTrends(trendsData);
        setSentiment(sentimentData);
      } catch (error) {
        console.error("Dashboard data load failed:", error);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  return (
    <div className="space-y-8 pb-12">
      {/* Search and Header Section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground mb-1">Intelligence Dashboard</h1>
          <p className="text-muted text-sm">Real-time analysis of dynamic trends and event distributions.</p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted" />
            <input
              type="text"
              placeholder="Search intelligence..."
              className="glass pl-10 pr-4 py-2 rounded-xl text-sm border-border/50 focus:border-primary outline-none transition-all w-64"
            />
          </div>
          <button className="glass p-2 rounded-xl border-border/50 hover:bg-secondary transition-colors relative">
            <Bell className="h-5 w-5 text-muted" />
            <span className={`absolute top-2 right-2 h-2 w-2 rounded-full border-2 border-background ${connectionState === "connected" ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`} />
          </button>
        </div>
      </div>

      {/* KPI Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Total Intelligence Units"
          value={((stats?.total_articles || 0) + totalReceived).toLocaleString()}
          icon={Activity}
          isLoading={loading}
          trend={{ value: 12, isUp: true }}
        />
        <KPICard
          title="Active Themes"
          value={stats?.topics_count || "0"}
          icon={BarChartIcon}
          isLoading={loading}
        />
        <KPICard
          title="Avg Sentiment"
          value={liveAvgSentiment !== 0 ? liveAvgSentiment.toFixed(2) : (stats?.sentiment_avg?.toFixed(2) || "0.00")}
          icon={MessageSquare}
          isLoading={loading}
          trend={{ value: 4, isUp: (liveAvgSentiment || stats?.sentiment_avg) > 0 }}
        />
        <KPICard
          title="Hybrid Model"
          value="ARIMA+BiLSTM"
          icon={TrendingUp}
          isLoading={loading}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Chart Area */}
        <div className="lg:col-span-2 space-y-8">
          <TrendChart data={trends} topics={topics} isLoading={loading} />

          {/* Real Sentiment Chart */}
          <SentimentChart data={sentiment} isLoading={loading} />
        </div>

        {/* Sidebar Analytics */}
        <div className="space-y-6">
          <div className="glass rounded-3xl p-6 border border-border/50">
            <h3 className="text-lg font-bold mb-6">Discovered Themes</h3>
            <TopicList
              topics={topics}
              isLoading={loading}
              selectedId={selectedTopicId}
              onSelect={setSelectedTopicId}
            />
          </div>

          <div className="glass rounded-3xl p-6 border border-border/50 bg-primary/5">
            <h4 className="text-sm font-bold text-primary uppercase tracking-widest mb-4">Phase 3 Engine</h4>
            <p className="text-xs text-muted leading-relaxed">
              Hybrid ARIMA + BiLSTM forecasting is active for <strong>{topics.find(t => t.id === selectedTopicId)?.name.split(':')[0]}</strong>.
              The model combines linear trend extraction with non-linear residual correction for optimal prediction accuracy.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
