"use client";

import React from "react";
import { PortfolioSummary, PositionItem } from "@/lib/api";
import { DollarSign, Layers, TrendingUp, TrendingDown, PieChart } from "lucide-react";

interface PortfolioViewProps {
  portfolio: PortfolioSummary | null;
  positions: PositionItem[];
}

export const PortfolioView: React.FC<PortfolioViewProps> = ({
  portfolio,
  positions,
}) => {
  const equity = portfolio?.equity ?? 100000;
  const cash = portfolio?.cash ?? 100000;
  const buyingPower = portfolio?.buying_power ?? 100000;
  const grossExposure = portfolio?.gross_exposure ?? 0;
  const exposurePct = portfolio?.exposure_pct ?? 0;

  const totalUnrealizedPnl = positions.reduce((acc, p) => acc + p.unrealized_pnl, 0);
  const isProfit = totalUnrealizedPnl >= 0;

  // Colors for asset allocation bar
  const palette = [
    "#06B6D4", "#3B82F6", "#8B5CF6", "#EC4899", "#10B981",
    "#F59E0B", "#6366F1", "#14B8A6", "#F97316", "#84CC16"
  ];

  return (
    <div className="space-y-6">
      {/* Portfolio Financial Header */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-4">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Total Account Equity
          </span>
          <div className="text-2xl font-bold font-mono text-white mt-2">
            ${equity.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </div>
          <span className="text-[10px] text-cyan-400 font-medium">Reconciled with Paper Broker</span>
        </div>

        <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-4">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Cash Balance
          </span>
          <div className="text-2xl font-bold font-mono text-slate-200 mt-2">
            ${cash.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </div>
          <span className="text-[10px] text-slate-400">Available Buying Power: ${buyingPower.toLocaleString()}</span>
        </div>

        <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-4">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Invested Market Value
          </span>
          <div className="text-2xl font-bold font-mono text-cyan-400 mt-2">
            ${grossExposure.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </div>
          <span className="text-[10px] text-slate-400">{exposurePct.toFixed(1)}% of Portfolio Equity</span>
        </div>

        <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-4">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Unrealized P&L
          </span>
          <div className={`text-2xl font-bold font-mono mt-2 ${isProfit ? "text-emerald-400" : "text-rose-400"}`}>
            {isProfit ? "+" : ""}${totalUnrealizedPnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </div>
          <span className="text-[10px] text-slate-400">{positions.length} Active Positions</span>
        </div>
      </div>

      {/* Asset Allocation Breakdown Bar */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
            <PieChart className="w-4 h-4 text-cyan-400" />
            <span>Asset Allocation & Capital Exposure</span>
          </h3>
          <span className="text-xs font-mono text-slate-400">
            Cash: {((cash / equity) * 100).toFixed(1)}% &bull; Stocks: {exposurePct.toFixed(1)}%
          </span>
        </div>

        {/* Stacked allocation progress bar */}
        <div className="h-4 w-full rounded-full overflow-hidden bg-[#1A2338] flex">
          {positions.map((pos, idx) => (
            <div
              key={pos.symbol}
              style={{
                width: `${pos.exposure_pct}%`,
                backgroundColor: palette[idx % palette.length],
              }}
              className="h-full transition-all"
              title={`${pos.symbol}: ${pos.exposure_pct.toFixed(1)}%`}
            />
          ))}
          <div
            style={{ width: `${Math.max(0, 100 - exposurePct)}%` }}
            className="h-full bg-slate-700/60"
            title={`Cash: ${((cash / equity) * 100).toFixed(1)}%`}
          />
        </div>

        {/* Allocation Legend */}
        <div className="flex flex-wrap gap-4 mt-3 pt-3 border-t border-[#1E293B]/60 text-xs font-mono">
          {positions.map((pos, idx) => (
            <div key={pos.symbol} className="flex items-center gap-1.5">
              <span
                className="w-2.5 h-2.5 rounded-sm"
                style={{ backgroundColor: palette[idx % palette.length] }}
              />
              <span className="text-white font-bold">{pos.symbol}</span>
              <span className="text-slate-400">({pos.exposure_pct.toFixed(1)}%)</span>
            </div>
          ))}
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-slate-700" />
            <span className="text-slate-300">USD Cash</span>
            <span className="text-slate-400">({((cash / equity) * 100).toFixed(1)}%)</span>
          </div>
        </div>
      </div>

      {/* Full Holdings Table */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 shadow-lg">
        <h3 className="text-sm font-bold text-white tracking-tight mb-4">
          Detailed Open Holdings & Performance
        </h3>

        {positions.length === 0 ? (
          <div className="py-12 text-center text-slate-500 text-xs">
            No active positions currently held in paper trading portfolio.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-[#1E293B] text-slate-400">
                  <th className="py-2.5 px-3">Symbol</th>
                  <th className="py-2.5 px-3 text-right">Shares</th>
                  <th className="py-2.5 px-3 text-right">Avg Entry Price</th>
                  <th className="py-2.5 px-3 text-right">Current Price</th>
                  <th className="py-2.5 px-3 text-right">Market Value</th>
                  <th className="py-2.5 px-3 text-right">Unrealized P&L ($)</th>
                  <th className="py-2.5 px-3 text-right">Return (%)</th>
                  <th className="py-2.5 px-3 text-right">Exposure</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E293B]/60 text-slate-300">
                {positions.map((pos) => {
                  const isPosProfit = pos.unrealized_pnl >= 0;
                  return (
                    <tr key={pos.symbol} className="hover:bg-[#1A2338]/40">
                      <td className="py-2.5 px-3 font-bold text-white">
                        {pos.symbol}
                      </td>
                      <td className="py-2.5 px-3 text-right text-slate-200">
                        {pos.shares}
                      </td>
                      <td className="py-2.5 px-3 text-right text-slate-400">
                        ${pos.avg_entry_price.toFixed(2)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold text-white">
                        ${pos.current_price.toFixed(2)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold font-mono text-cyan-300">
                        ${pos.market_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                      </td>
                      <td className={`py-2.5 px-3 text-right font-bold ${isPosProfit ? "text-emerald-400" : "text-rose-400"}`}>
                        {isPosProfit ? "+" : ""}${pos.unrealized_pnl.toFixed(2)}
                      </td>
                      <td className={`py-2.5 px-3 text-right font-bold ${isPosProfit ? "text-emerald-400" : "text-rose-400"}`}>
                        {isPosProfit ? "+" : ""}{pos.unrealized_pnl_pct.toFixed(2)}%
                      </td>
                      <td className="py-2.5 px-3 text-right text-slate-400">
                        {pos.exposure_pct.toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
