"use client";

import React from "react";
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  ShieldCheck,
  AlertTriangle,
  Layers,
  Cpu,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";
import {
  PortfolioSummary,
  PositionItem,
  EquityPoint,
  PredictionItem,
  SignalItem,
  RiskStatus,
} from "@/lib/api";
import { EquityChart } from "../EquityChart";

interface DashboardViewProps {
  portfolio: PortfolioSummary | null;
  positions: PositionItem[];
  equityHistory: EquityPoint[];
  predictions: PredictionItem[];
  signals: SignalItem[];
  riskStatus: RiskStatus | null;
  onNavigateTab: (tab: string) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  portfolio,
  positions,
  equityHistory,
  predictions,
  signals,
  riskStatus,
  onNavigateTab,
}) => {
  const equity = portfolio?.equity ?? 100000;
  const cash = portfolio?.cash ?? 100000;
  const buyingPower = portfolio?.buying_power ?? 100000;
  const dayPnl = portfolio?.day_pnl ?? 0;
  const dayPnlPct = portfolio?.day_pnl_pct ?? 0;
  const totalPnl = portfolio?.total_pnl ?? 0;
  const exposurePct = portfolio?.exposure_pct ?? 0;

  const isDayPositive = dayPnl >= 0;
  const isTotalPositive = totalPnl >= 0;

  return (
    <div className="space-y-6">
      {/* Top Warning Banner if Circuit Breaker Tripped */}
      {riskStatus?.circuit_breaker_tripped && (
        <div className="bg-rose-500/15 border border-rose-500/40 rounded-xl p-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
            <div>
              <h4 className="text-sm font-bold text-white">
                Circuit Breaker Active
              </h4>
              <p className="text-xs text-rose-200">
                {riskStatus.trip_reason ||
                  "Risk engine tripped emergency safeguard. New paper orders are blocked."}
              </p>
            </div>
          </div>
          <button
            onClick={() => onNavigateTab("risk")}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-600 hover:bg-rose-500 text-white shrink-0"
          >
            Review Safeguards
          </button>
        </div>
      )}

      {/* Primary KPI Cards Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3.5">
        {/* Total Equity */}
        <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-4 flex flex-col justify-between">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Total Equity
          </span>
          <div className="my-2">
            <span className="text-xl lg:text-2xl font-bold font-mono text-white tracking-tight">
              ${equity.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </span>
          </div>
          <span className="text-[10px] text-cyan-400 font-medium">
            Paper Account
          </span>
        </div>

        {/* Buying Power */}
        <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-4 flex flex-col justify-between">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Buying Power
          </span>
          <div className="my-2">
            <span className="text-xl lg:text-2xl font-bold font-mono text-white tracking-tight">
              ${buyingPower.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </span>
          </div>
          <span className="text-[10px] text-slate-400">Available Capital</span>
        </div>

        {/* Cash Balance */}
        <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-4 flex flex-col justify-between">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Cash Balance
          </span>
          <div className="my-2">
            <span className="text-xl lg:text-2xl font-bold font-mono text-slate-200 tracking-tight">
              ${cash.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </span>
          </div>
          <span className="text-[10px] text-slate-400">Unallocated USD</span>
        </div>

        {/* Day P&L */}
        <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-4 flex flex-col justify-between">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Daily P&L
          </span>
          <div className="my-2 flex items-center gap-1.5">
            {isDayPositive ? (
              <ArrowUpRight className="w-4 h-4 text-emerald-400" />
            ) : (
              <ArrowDownRight className="w-4 h-4 text-rose-400" />
            )}
            <span
              className={`text-xl lg:text-2xl font-bold font-mono tracking-tight ${
                isDayPositive ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {isDayPositive ? "+" : ""}
              ${dayPnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </span>
          </div>
          <span
            className={`text-[10px] font-semibold ${
              isDayPositive ? "text-emerald-400" : "text-rose-400"
            }`}
          >
            {isDayPositive ? "+" : ""}
            {dayPnlPct.toFixed(2)}%
          </span>
        </div>

        {/* Total P&L */}
        <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-4 flex flex-col justify-between">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Total Return
          </span>
          <div className="my-2 flex items-center gap-1.5">
            {isTotalPositive ? (
              <ArrowUpRight className="w-4 h-4 text-emerald-400" />
            ) : (
              <ArrowDownRight className="w-4 h-4 text-rose-400" />
            )}
            <span
              className={`text-xl lg:text-2xl font-bold font-mono tracking-tight ${
                isTotalPositive ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {isTotalPositive ? "+" : ""}
              ${totalPnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </span>
          </div>
          <span className="text-[10px] text-slate-400">Since inception</span>
        </div>

        {/* Gross Exposure */}
        <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-4 flex flex-col justify-between">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Exposure
          </span>
          <div className="my-2">
            <span className="text-xl lg:text-2xl font-bold font-mono text-cyan-400 tracking-tight">
              {exposurePct.toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-[#1A2338] rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-cyan-500 h-full rounded-full transition-all"
              style={{ width: `${Math.min(100, exposurePct)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Main Chart and Positions Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Equity Timeline Card */}
        <div className="lg:col-span-2 bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold text-white tracking-tight">
                Portfolio Performance Timeline
              </h3>
              <p className="text-xs text-slate-400">
                Mark-to-market equity balance tracked per trading cycle
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
              <span>Real-Time Reconciled</span>
            </div>
          </div>
          <EquityChart points={equityHistory} />
        </div>

        {/* Open Positions Summary */}
        <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 flex flex-col justify-between shadow-lg">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-bold text-white tracking-tight">
                Open Holdings ({positions.length})
              </h3>
              <button
                onClick={() => onNavigateTab("portfolio")}
                className="text-xs text-cyan-400 hover:text-cyan-300 font-medium"
              >
                View Details &rarr;
              </button>
            </div>

            {positions.length === 0 ? (
              <div className="py-12 text-center text-slate-500 text-xs">
                No active stock holdings. Agent is in cash.
              </div>
            ) : (
              <div className="space-y-3">
                {positions.slice(0, 5).map((pos) => {
                  const isPosProfit = pos.unrealized_pnl >= 0;
                  return (
                    <div
                      key={pos.symbol}
                      className="flex items-center justify-between p-2.5 rounded-lg bg-[#1A2338]/60 border border-[#2D3A58]/40 text-xs"
                    >
                      <div>
                        <span className="font-bold text-white font-mono">
                          {pos.symbol}
                        </span>
                        <div className="text-[11px] text-slate-400">
                          {pos.shares} shs @ ${pos.avg_entry_price.toFixed(2)}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-mono text-white font-semibold">
                          ${pos.market_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </div>
                        <div
                          className={`font-mono font-medium ${
                            isPosProfit ? "text-emerald-400" : "text-rose-400"
                          }`}
                        >
                          {isPosProfit ? "+" : ""}
                          {pos.unrealized_pnl_pct.toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-[#1E293B] text-[11px] text-slate-500 flex items-center justify-between">
            <span>Max Single Position: 10%</span>
            <span>Max Positions: 10</span>
          </div>
        </div>
      </div>

      {/* Latest Model Signals & Predictions Glance */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">
              Latest Machine Learning Predictions & Signals
            </h3>
            <p className="text-xs text-slate-400">
              Random Forest probability estimates evaluated across universe
            </p>
          </div>
          <button
            onClick={() => onNavigateTab("predictions")}
            className="text-xs text-cyan-400 hover:text-cyan-300 font-medium"
          >
            All Predictions &rarr;
          </button>
        </div>

        {predictions.length === 0 ? (
          <div className="py-8 text-center text-slate-500 text-xs">
            No predictions generated yet. Click "Run Cycle" to evaluate strategy.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3.5">
            {predictions.slice(0, 4).map((p) => {
              const matchedSig = signals.find((s) => s.symbol === p.symbol);
              const sigColor =
                matchedSig?.signal === "BUY"
                  ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/30"
                  : matchedSig?.signal === "EXIT"
                  ? "text-rose-400 bg-rose-500/10 border-rose-500/30"
                  : "text-slate-400 bg-slate-500/10 border-slate-500/30";

              return (
                <div
                  key={p.symbol + p.timestamp}
                  className="bg-[#1A2338]/60 border border-[#2D3A58]/50 rounded-lg p-3.5 flex flex-col justify-between"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-white font-mono text-sm">
                      {p.symbol}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold border ${sigColor}`}
                    >
                      {matchedSig?.signal || "HOLD"}
                    </span>
                  </div>

                  <div className="space-y-1 my-2 text-[11px] font-mono">
                    <div className="flex justify-between text-slate-400">
                      <span>P(UP)</span>
                      <span className="text-emerald-400 font-bold">
                        {(p.prob_up * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>P(NEUT)</span>
                      <span className="text-slate-300">
                        {(p.prob_neutral * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>P(DOWN)</span>
                      <span className="text-rose-400 font-bold">
                        {(p.prob_down * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="text-[10px] text-slate-500 truncate pt-2 border-t border-[#2D3A58]/50">
                    Model: {p.model_version}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
