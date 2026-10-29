"use client";

import React, { useState } from "react";
import { BarItem } from "@/lib/api";

interface PriceChartProps {
  symbol: string;
  bars: BarItem[];
}

export const PriceChart: React.FC<PriceChartProps> = ({ symbol, bars }) => {
  const [hoveredBar, setHoveredBar] = useState<BarItem | null>(null);

  if (!bars || bars.length === 0) {
    return (
      <div className="h-72 flex items-center justify-center text-slate-500 text-sm">
        No price bars available for {symbol}.
      </div>
    );
  }

  const width = 800;
  const height = 300;
  const padding = { top: 20, right: 40, bottom: 40, left: 60 };

  const chartW = width - padding.left - padding.right;
  const priceH = height * 0.7 - padding.top;
  const volumeH = height * 0.25;
  const volumeY = height - padding.bottom - volumeH;

  const highs = bars.map((b) => b.high);
  const lows = bars.map((b) => b.low);
  const volumes = bars.map((b) => b.volume);

  const minPrice = Math.min(...lows) * 0.995;
  const maxPrice = Math.max(...highs) * 1.005;
  const priceRange = maxPrice - minPrice || 1.0;
  const maxVolume = Math.max(...volumes) || 1.0;

  const barCount = bars.length;
  const barSpacing = chartW / Math.max(1, barCount);
  const candleW = Math.max(2, barSpacing * 0.7);

  return (
    <div className="relative w-full overflow-hidden select-none">
      {/* Hover Information Banner */}
      <div className="flex items-center justify-between text-xs font-mono px-2 py-1 mb-2 bg-[#1A2338]/60 rounded-md border border-[#2D3A58]/50">
        <div className="text-slate-300">
          <span className="font-bold text-cyan-400 mr-2">{symbol}</span>
          <span>1D INTERVAL</span>
        </div>
        {hoveredBar ? (
          <div className="flex items-center gap-3 text-slate-300">
            <span>
              O:{" "}
              <b className="text-white">
                ${hoveredBar.open.toFixed(2)}
              </b>
            </span>
            <span>
              H:{" "}
              <b className="text-white">
                ${hoveredBar.high.toFixed(2)}
              </b>
            </span>
            <span>
              L:{" "}
              <b className="text-white">
                ${hoveredBar.low.toFixed(2)}
              </b>
            </span>
            <span>
              C:{" "}
              <b
                className={
                  hoveredBar.close >= hoveredBar.open
                    ? "text-emerald-400"
                    : "text-rose-400"
                }
              >
                ${hoveredBar.close.toFixed(2)}
              </b>
            </span>
            <span>
              Vol:{" "}
              <b className="text-white">
                {(hoveredBar.volume / 1000).toFixed(0)}k
              </b>
            </span>
          </div>
        ) : (
          <div className="text-slate-500">Hover over candles for OHLC details</div>
        )}
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
        {/* Horizontal price grid lines */}
        {[0, 0.33, 0.66, 1.0].map((ratio) => {
          const y = padding.top + priceH * (1 - ratio);
          const val = minPrice + priceRange * ratio;
          return (
            <g key={ratio}>
              <line
                x1={padding.left}
                y1={y}
                x2={padding.left + chartW}
                y2={y}
                stroke="#1E293B"
                strokeDasharray="4 4"
              />
              <text
                x={padding.left - 10}
                y={y + 3}
                fill="#64748B"
                fontSize="10"
                fontFamily="monospace"
                textAnchor="end"
              >
                ${val.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* Volume separation line */}
        <line
          x1={padding.left}
          y1={volumeY - 5}
          x2={padding.left + chartW}
          y2={volumeY - 5}
          stroke="#2D3A58"
        />

        {/* Render Candles & Volume Bars */}
        {bars.map((b, i) => {
          const x = padding.left + i * barSpacing + barSpacing / 2;
          const isUp = b.close >= b.open;
          const color = isUp ? "#10B981" : "#EF4444";

          // Y coordinates for Price
          const yHigh = padding.top + priceH - ((b.high - minPrice) / priceRange) * priceH;
          const yLow = padding.top + priceH - ((b.low - minPrice) / priceRange) * priceH;
          const yOpen = padding.top + priceH - ((b.open - minPrice) / priceRange) * priceH;
          const yClose = padding.top + priceH - ((b.close - minPrice) / priceRange) * priceH;

          const topBody = Math.min(yOpen, yClose);
          const bodyH = Math.max(2, Math.abs(yClose - yOpen));

          // Volume Bar
          const volBarH = (b.volume / maxVolume) * volumeH;
          const volY = height - padding.bottom - volBarH;

          return (
            <g
              key={i}
              className="cursor-pointer"
              onMouseEnter={() => setHoveredBar(b)}
              onMouseLeave={() => setHoveredBar(null)}
            >
              {/* Wick */}
              <line
                x1={x}
                y1={yHigh}
                x2={x}
                y2={yLow}
                stroke={color}
                strokeWidth="1.5"
              />

              {/* Candle Body */}
              <rect
                x={x - candleW / 2}
                y={topBody}
                width={candleW}
                height={bodyH}
                fill={color}
                rx="1"
              />

              {/* Volume Bar */}
              <rect
                x={x - candleW / 2}
                y={volY}
                width={candleW}
                height={volBarH}
                fill={color}
                opacity="0.35"
                rx="1"
              />
            </g>
          );
        })}
      </svg>
    </div>
  );
};
