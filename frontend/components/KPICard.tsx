"use client";

import { ReactNode } from "react";
import { LucideIcon } from "lucide-react";
import { motion } from "framer-motion";

interface KPICardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  trend?: {
    value: number;
    isUp: boolean;
  };
  isLoading?: boolean;
}

export default function KPICard({ title, value, icon: Icon, trend, isLoading }: KPICardProps) {
  if (isLoading) {
    return (
      <div className="glass rounded-3xl p-6 relative overflow-hidden h-40">
        <div className="shimmer absolute inset-0" />
      </div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -5 }}
      transition={{ duration: 0.3 }}
      className="glass group rounded-3xl p-6 relative overflow-hidden border border-border/50 hover:border-primary/30 transition-colors"
    >
      <div className="absolute -right-4 -top-4 h-24 w-24 rounded-full bg-primary/5 blur-2xl group-hover:bg-primary/10 transition-colors" />
      
      <div className="flex items-center justify-between mb-4">
        <div className="p-3 rounded-2xl bg-secondary/50 text-primary border border-border/20 group-hover:bg-primary group-hover:text-primary-foreground transition-all duration-300">
          <Icon className="h-6 w-6" />
        </div>
        {trend && (
          <div className={`flex items-center text-xs font-bold px-2.5 py-1 rounded-full ${
            trend.isUp ? "bg-emerald-500/10 text-emerald-500" : "bg-rose-500/10 text-rose-500"
          }`}>
            {trend.isUp ? "+" : "-"}{Math.abs(trend.value)}%
          </div>
        )}
      </div>

      <div>
        <h3 className="text-sm font-medium text-muted mb-1">{title}</h3>
        <p className="text-3xl font-bold tracking-tight text-foreground">{value}</p>
      </div>
    </motion.div>
  );
}
