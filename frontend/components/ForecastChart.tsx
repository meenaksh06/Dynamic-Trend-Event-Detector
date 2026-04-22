"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceArea
} from "recharts";
import { motion } from "framer-motion";

interface ForecastChartProps {
  historicalData: any[];
  forecastData: any[];
  topicName: string;
  isLoading?: boolean;
}

export default function ForecastChart({ historicalData, forecastData, topicName, isLoading }: ForecastChartProps) {
  if (isLoading) {
    return (
      <div className="glass rounded-3xl p-8 h-[400px] relative overflow-hidden">
        <div className="shimmer absolute inset-0" />
      </div>
    );
  }

  // Combine data for the chart
  const lastHistorical = historicalData[historicalData.length - 1];
  const combinedData = [...historicalData.map(d => ({ ...d, type: 'historical' }))];
  
  if (forecastData.length > 0) {
    // Add the bridge point
    combinedData.push({ 
      date: forecastData[0].date, 
      value: lastHistorical.value, 
      type: 'bridge' 
    });
    combinedData.push(...forecastData.map(d => ({ ...d, type: 'forecast' })));
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-3xl p-6 border border-border/50"
    >
      <div className="mb-8">
        <h3 className="text-xl font-bold text-foreground">Temporal Projection: {topicName}</h3>
        <p className="text-sm text-muted">LSTM-based 6-month forecasting model</p>
      </div>

      <div className="h-[400px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={combinedData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis 
              dataKey="date" 
              stroke="#64748b" 
              fontSize={10} 
              tickLine={false} 
              axisLine={false}
              tickFormatter={(str) => {
                const parts = str.split('-');
                return parts.length > 1 ? `${parts[1]}/${parts[0].slice(2)}` : str;
              }}
            />
            <YAxis 
              stroke="#64748b" 
              fontSize={10} 
              tickLine={false} 
              axisLine={false} 
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: "rgba(15, 23, 42, 0.9)", 
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "12px",
                backdropFilter: "blur(10px)"
              }}
            />
            <Legend verticalAlign="top" height={36}/>
            
            <Line
              type="monotone"
              dataKey="value"
              name="Historical"
              stroke="#38bdf8"
              strokeWidth={3}
              dot={{ r: 4, fill: "#38bdf8", strokeWidth: 0 }}
              activeDot={{ r: 6 }}
              connectNulls
              data={combinedData.filter(d => d.type === 'historical' || d.type === 'bridge')}
            />
            
            <Line
              type="monotone"
              dataKey="value"
              name="Forecast"
              stroke="#818cf8"
              strokeWidth={3}
              strokeDasharray="5 5"
              dot={{ r: 4, fill: "#818cf8", strokeWidth: 0 }}
              activeDot={{ r: 6 }}
              connectNulls
              data={combinedData.filter(d => d.type === 'forecast' || d.type === 'bridge')}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
