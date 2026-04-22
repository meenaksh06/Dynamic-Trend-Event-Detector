"use client";

import { motion } from "framer-motion";
import { Tag, ArrowRight } from "lucide-react";

interface Topic {
  id: number;
  name: string;
  count: number;
  keywords: string[];
}

interface TopicListProps {
  topics: Topic[];
  selectedId?: number;
  onSelect: (id: number) => void;
  isLoading?: boolean;
}

export default function TopicList({ topics, selectedId, onSelect, isLoading }: TopicListProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="glass rounded-2xl p-4 h-20 relative overflow-hidden">
            <div className="shimmer absolute inset-0" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {topics.map((topic, index) => (
        <motion.div
          key={topic.id}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.05 }}
          onClick={() => onSelect(topic.id)}
          className={`glass group cursor-pointer flex items-center justify-between p-4 rounded-2xl border transition-all duration-300 ${
            selectedId === topic.id 
              ? "border-primary bg-primary/5 shadow-lg shadow-primary/10" 
              : "border-border/50 hover:border-border hover:bg-secondary/30"
          }`}
        >
          <div className="flex items-center gap-4">
            <div className={`h-10 w-10 flex items-center justify-center rounded-xl transition-colors ${
              selectedId === topic.id ? "bg-primary text-primary-foreground" : "bg-secondary text-muted group-hover:text-foreground"
            }`}>
              <Tag className="h-5 w-5" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-foreground group-hover:text-primary transition-colors">
                {topic.name}
              </h4>
              <div className="flex flex-wrap gap-1 mt-1">
                {topic.keywords.slice(0, 3).map((word) => (
                  <span key={word} className="text-[10px] px-1.5 py-0.5 rounded-md bg-secondary/50 text-muted border border-border/10">
                    {word}
                  </span>
                ))}
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="text-right">
              <p className="text-xs font-bold text-foreground">{topic.count}</p>
              <p className="text-[10px] text-muted uppercase">Articles</p>
            </div>
            <ArrowRight className={`h-4 w-4 transition-transform duration-300 ${
              selectedId === topic.id ? "translate-x-1 text-primary" : "text-muted group-hover:translate-x-1 group-hover:text-foreground"
            }`} />
          </div>
        </motion.div>
      ))}
    </div>
  );
}
