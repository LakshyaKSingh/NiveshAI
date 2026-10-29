"use client";

import React, { useState } from "react";
import { EquityPoint } from "@/lib/api";

interface EquityChartProps {
  points: EquityPoint[];
}

export const EquityChart: React.FC<EquityChartProps> = ({ points }) => {
  const [hoveredPoint, setHoveredPoint] = useState<EquityPoint | null>(null);

  if (!points || points.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-500 text-sm">
        No equity history points recorded yet.
      </div>
    );
  }

  const values = points.map((p) => p.equity);
  const minVal = Math.min(...values) * 0.998;
  const maxVal = Math.max(...values) * 1.002;
  const range = maxVal - minVal || 1.0;

  const width = 800;
  const height = 260;
  const padding = { top: 20, right: 30, bottom: 30, left: 70 };

  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  // Generate SVG path points
  const coords = points.map((p, idx) => {
    const x =
      padding.left + (idx / Math.max(1, points.length - 1)) * chartW;
    const y =
      padding.top + chartH - ((p.equity - minVal) / range) * chartH;
    return { x, y, point: p };
  });

  const lineD = coords.reduce(
    (acc, curr, idx) => `${acc} ${idx === 0 ? "M" : "L"} ${curr.x} ${curr.y}`,
    ""
  );

  const areaD = `${lineD} L ${padding.left + chartW} ${
    padding.top + chartH
  } L ${padding.left} ${padding.top + chartH} Z`;

  // Determine profit or loss trend
  const firstVal = points[0].equity;
  const lastVal = points[points.length - 1].equity;
  const isProfit = lastVal >= firstVal;
  const strokeColor = isProfit ? "#10B981" : "#EF4444";
  const gradientId = isProfit ? "equityGradientGreen" : "equityGradientRed";

  return (
    <div className="relative w-full overflow-hidden">
      {/* Chart Tooltip Overlay */}
      {hoveredPoint && (
        <div className="absolute top-2 right-4 bg-[#1A2338] border border-[#2D3A58] rounded-lg px-3 py-1.5 text-xs shadow-xl pointer-events-none">
          <div className="text-slate-400">
            {new Date(hoveredPoint.timestamp).toLocaleString()}
          </div>
          <div className="text-white font-bold font-mono text-sm">
            ${hoveredPoint.equity.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </div>
          <div className="text-slate-400 text-[11px]">
            Cash: ${hoveredPoint.cash.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </div>
        </div>
      )}

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto select-none"
      >
        <defs>
          <linearGradient id="equityGradientGreen" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#10B981" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#10B981" stopOpacity="0.0" />
          </linearGradient>
          <linearGradient id="equityGradientRed" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#EF4444" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#EF4444" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* Horizontal grid lines */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
          const y = padding.top + chartH * (1 - ratio);
          const val = minVal + range * ratio;
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
                y={y + 4}
                fill="#64748B"
                fontSize="10"
                fontFamily="monospace"
                textAnchor="end"
              >
                ${val.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </text>
            </g>
          );
        })}

        {/* Area fill */}
        <path d={areaD} fill={`url(#${gradientId})`} />

        {/* Equity line */}
        <path
          d={lineD}
          fill="none"
          stroke={strokeColor}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Interactive hover points */}
        {coords.map((c, i) => (
          <circle
            key={i}
            cx={c.x}
            cy={c.y}
            r="4"
            className="cursor-pointer transition-all hover:r-6"
            fill={strokeColor}
            stroke="#0D1424"
            strokeWidth="2"
            onMouseEnter={() => setHoveredPoint(c.point)}
            onMouseLeave={() => setHoveredPoint(null)}
          />
        ))}
      </svg>
    </div>
  );
};
