"use client";

import { useEffect, useState } from "react";
import { Newspaper, ExternalLink, Filter, Calendar } from "lucide-react";
import TopicList from "@/components/TopicList";
import { fetchTopics, fetchArticles } from "@/lib/api";
import { useLiveContext } from "@/components/WebSocketProvider";
import { motion, AnimatePresence } from "framer-motion";

export default function EventExplorer() {
  const [topics, setTopics] = useState<any[]>([]);
  const [articles, setArticles] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingArticles, setLoadingArticles] = useState(false);
  const [selectedTopicId, setSelectedTopicId] = useState<number>(0);
  const { lastEvent, connectionState } = useLiveContext();

  useEffect(() => {
    async function loadInitialData() {
      try {
        const topicsData = await fetchTopics();
        setTopics(topicsData);
        if (topicsData.length > 0) {
          loadArticles(topicsData[0].id);
        }
      } catch (error) {
        console.error("Initial load failed:", error);
      } finally {
        setLoading(false);
      }
    }
    loadInitialData();
  }, []);

  async function loadArticles(topicId: number) {
    setLoadingArticles(true);
    try {
      const articlesData = await fetchArticles(topicId, 20);
      setArticles(articlesData);
      setSelectedTopicId(topicId);
    } catch (error) {
      console.error("Articles load failed:", error);
    } finally {
      setLoadingArticles(false);
    }
  }

  // Prepend live events that match the current topic
  useEffect(() => {
    if (lastEvent && lastEvent.topic_id === selectedTopicId) {
      setArticles(prev => {
        // Prevent duplicates based on headline and timestamp just in case
        if (prev.some(a => a.headline === lastEvent.headline && a.date === lastEvent.timestamp)) {
          return prev;
        }
        return [{
          date: lastEvent.timestamp,
          headline: lastEvent.headline,
          category: lastEvent.category,
          sentiment_compound: lastEvent.sentiment_compound,
          link: lastEvent.link,
          // Custom flag to highlight it's a live event
          isLive: true,
          eventType: lastEvent.event_type
        }, ...prev];
      });
    }
  }, [lastEvent, selectedTopicId]);

  const selectedTopic = topics.find(t => t.id === selectedTopicId);

  return (
    <div className="space-y-8 pb-12">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="h-12 w-12 rounded-2xl bg-primary/20 flex items-center justify-center text-primary">
            <Newspaper className="h-7 w-7" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-foreground">Event Explorer</h1>
            <p className="text-muted text-sm">Deep dive into individual news events categorized by theme.</p>
          </div>
        </div>
        
        {connectionState === "connected" && (
          <div className="glass px-4 py-1.5 rounded-full border border-emerald-500/30 flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] text-emerald-400 font-bold tracking-widest uppercase">Live Injection Active</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Topic Filters */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass rounded-3xl p-6 border border-border/50">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-bold">Categories</h3>
              <Filter className="h-4 w-4 text-muted" />
            </div>
            <TopicList 
              topics={topics} 
              isLoading={loading} 
              selectedId={selectedTopicId}
              onSelect={loadArticles}
            />
          </div>
        </div>

        {/* Article Feed */}
        <div className="lg:col-span-3 space-y-4">
          <div className="flex items-center justify-between px-2">
            <h2 className="text-xl font-bold flex items-center gap-2">
              {selectedTopic?.name || "Topic"} 
              <span className="text-xs font-normal text-muted bg-secondary px-2 py-0.5 rounded-full">
                Recent Reports
              </span>
            </h2>
          </div>

          <div className="space-y-4">
            {loadingArticles ? (
              [1, 2, 3, 4, 5].map(i => (
                <div key={i} className="glass rounded-2xl p-6 h-32 relative overflow-hidden">
                  <div className="shimmer absolute inset-0" />
                </div>
              ))
            ) : (
              <AnimatePresence mode="popLayout">
                {articles.map((article, index) => (
                  <motion.div
                    key={article.link + index}
                    layout
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ duration: 0.2, delay: index * 0.05 }}
                    className={`glass group rounded-2xl p-6 border ${article.isLive ? 'border-emerald-500/50 shadow-lg shadow-emerald-500/10' : 'border-border/50'} hover:border-primary/30 transition-all duration-300`}
                  >
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          {article.isLive && (
                            <span className="text-[10px] uppercase tracking-widest font-bold text-emerald-400 px-2 py-0.5 rounded-md bg-emerald-500/10 flex items-center gap-1">
                              <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                              {article.eventType || "LIVE"}
                            </span>
                          )}
                          <span className="text-[10px] uppercase tracking-widest font-bold text-primary px-2 py-0.5 rounded-md bg-primary/10">
                            {article.category}
                          </span>
                          <span className="flex items-center text-[10px] text-muted font-medium italic">
                            <Calendar className="h-3 w-3 mr-1" />
                            {new Date(article.date).toLocaleDateString()}
                          </span>
                        </div>
                        <h3 className="text-lg font-bold text-foreground group-hover:text-primary transition-colors leading-tight max-w-2xl">
                          {article.headline}
                        </h3>
                        <div className="flex items-center gap-4">
                          <div className="flex items-center gap-2">
                            <div className={`h-1.5 w-1.5 rounded-full ${
                              article.sentiment_compound > 0.2 ? "bg-emerald-500" : 
                              article.sentiment_compound < -0.2 ? "bg-rose-500" : "bg-slate-500"
                            }`} />
                            <span className="text-xs text-muted">Sentiment: {article.sentiment_compound.toFixed(2)}</span>
                          </div>
                        </div>
                      </div>
                      
                      <a 
                        href={article.link} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-secondary hover:bg-primary hover:text-primary-foreground transition-all duration-300"
                      >
                        <ExternalLink className="h-5 w-5" />
                      </a>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            )}
            
            {articles.length === 0 && !loadingArticles && (
              <div className="text-center py-20 glass rounded-3xl border border-dashed border-border/50">
                <Newspaper className="h-12 w-12 text-muted mx-auto mb-4 opacity-20" />
                <p className="text-muted">No records found for this thematic category.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
