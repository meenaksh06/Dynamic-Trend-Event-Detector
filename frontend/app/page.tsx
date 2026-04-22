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
import { fetchStats, fetchTopics, fetchTrends } from "@/lib/api";
import { motion } from "framer-motion";

export default function Dashboard() {
  const [stats, setStats] = useState<any>(null);
  const [topics, setTopics] = useState<any[]>([]);
  const [trends, setTrends] = useState<any[]>([]);
  const [loading, setLoading] = useState(True);
  const [selectedTopicId, setSelectedTopicId] = useState<number>(0);

  useEffect(() => {
    async function loadData() {
      try {
        const [statsData, topicsData, trendsData] = await Promise.all([
          fetchStats(),
          fetchTopics(),
          fetchTrends()
        ]);
        setStats(statsData);
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
            <span className="absolute top-2 right-2 h-2 w-2 bg-rose-500 rounded-full border-2 border-background" />
          </button>
        </div>
      </div>

      {/* KPI Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard 
          title="Total Intelligence Units" 
          value={stats?.total_articles?.toLocaleString() || "0"} 
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
          value={stats?.sentiment_avg?.toFixed(2) || "0.00"} 
          icon={MessageSquare} 
          isLoading={loading}
          trend={{ value: 4, isUp: stats?.sentiment_avg > 0 }}
        />
        <KPICard 
          title="Users Monitored" 
          value="1,248" 
          icon={Users} 
          isLoading={loading}
          trend={{ value: 8, isUp: true }}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Chart Area */}
        <div className="lg:col-span-2 space-y-8">
          <TrendChart data={trends} topics={topics} isLoading={loading} />
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-3xl p-8 border border-border/50 h-[300px] flex items-center justify-center"
          >
            <div className="text-center">
              <TrendingUp className="h-12 w-12 text-primary mx-auto mb-4 opacity-50" />
              <h3 className="text-xl font-bold mb-2">Detailed Sentiment Analysis</h3>
              <p className="text-muted max-w-md">Our deep learning model identifies fine-grained emotional patterns within each detected theme.</p>
            </div>
          </motion.div>
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
            <h4 className="text-sm font-bold text-primary uppercase tracking-widest mb-4">Insights Engine</h4>
            <p className="text-xs text-muted leading-relaxed">
              LSTM forecasting indicates a 15% increase in volatility for <strong>{topics.find(t => t.id === selectedTopicId)?.name.split(':')[0]}</strong> over the next quarter. Recommendation: Increase monitoring frequency.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
