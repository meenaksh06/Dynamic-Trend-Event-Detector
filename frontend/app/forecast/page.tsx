"use client";

import { useEffect, useState } from "react";
import { BrainCircuit, Info, AlertTriangle } from "lucide-react";
import ForecastChart from "@/components/ForecastChart";
import TopicList from "@/components/TopicList";
import { fetchTopics, fetchTrends, fetchForecast } from "@/lib/api";
import { motion } from "framer-motion";

export default function ForecastPage() {
  const [topics, setTopics] = useState<any[]>([]);
  const [trends, setTrends] = useState<any[]>([]);
  const [forecast, setForecast] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingForecast, setLoadingForecast] = useState(false);
  const [selectedTopicId, setSelectedTopicId] = useState<number>(0);

  useEffect(() => {
    async function loadInitialData() {
      try {
        const [topicsData, trendsData] = await Promise.all([
          fetchTopics(),
          fetchTrends()
        ]);
        setTopics(topicsData);
        setTrends(trendsData);
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
      const forecastData = await fetchForecast(topicId);
      setForecast(forecastData);
      setSelectedTopicId(topicId);
    } catch (error) {
      console.error("Forecast load failed:", error);
    } finally {
      setLoadingForecast(false);
    }
  }

  const historicalValues = trends.map(t => ({
    date: t.date,
    value: t[`topic_${selectedTopicId}`]
  }));

  const selectedTopic = topics.find(t => t.id === selectedTopicId);

  return (
    <div className="space-y-8 pb-12">
      <div className="flex items-center gap-4">
        <div className="h-12 w-12 rounded-2xl bg-accent/20 flex items-center justify-center text-accent">
          <BrainCircuit className="h-7 w-7" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-foreground">Predictive Analytics</h1>
          <p className="text-muted text-sm">Long Short-Term Memory (LSTM) projections for emerging themes.</p>
        </div>
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
              Predictions are based on historical monthly frequency. High volatility events can lead to significant variances in mid-term accuracy.
            </p>
          </motion.div>
        </div>

        {/* Forecast Display */}
        <div className="lg:col-span-2 space-y-6">
          <ForecastChart 
            historicalData={historicalValues}
            forecastData={forecast}
            topicName={selectedTopic?.name || "Topic Analysis"}
            isLoading={loading || loadingForecast}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="glass rounded-2xl p-6 border border-border/50">
              <div className="flex items-center gap-3 mb-4">
                <Info className="h-5 w-5 text-primary" />
                <h4 className="font-bold text-foreground">Model Confidence</h4>
              </div>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted">Data Quality</span>
                    <span className="text-foreground font-bold">88%</span>
                  </div>
                  <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                    <div className="h-full bg-emerald-500 w-[88%]" />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted">Historical Fit</span>
                    <span className="text-foreground font-bold">92%</span>
                  </div>
                  <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                    <div className="h-full bg-primary w-[92%]" />
                  </div>
                </div>
              </div>
            </div>

            <div className="glass rounded-2xl p-6 border border-border/50">
              <h4 className="font-bold text-foreground mb-2">Technical Summary</h4>
              <p className="text-xs text-muted leading-relaxed">
                Network: Bi-directional LSTM<br/>
                Layers: 64 Units (Dense) + 32 Units (Drop)<br/>
                Optimizer: Adam (lr=0.001)<br/>
                Regularization: L2 (0.001)
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
