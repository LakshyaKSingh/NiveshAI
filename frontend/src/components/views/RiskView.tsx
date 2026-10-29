"use client";

import React from "react";
import { RiskStatus, RiskConfig, RiskEvent, AuditLog } from "@/lib/api";
import { ShieldCheck, AlertOctagon, ShieldAlert, History, Activity, CheckCircle2 } from "lucide-react";

interface RiskViewProps {
  riskStatus: RiskStatus | null;
  riskConfig: RiskConfig | null;
  riskEvents: RiskEvent[];
  auditLogs: AuditLog[];
}

export const RiskView: React.FC<RiskViewProps> = ({
  riskStatus,
  riskConfig,
  riskEvents,
  auditLogs,
}) => {
  const exposurePct = riskStatus?.gross_exposure_pct ?? 0;
  const maxExposurePct = riskStatus?.max_exposure_pct ?? 95;
  const positionsCount = riskStatus?.open_positions_count ?? 0;
  const maxPositions = riskStatus?.max_positions ?? 10;
  const drawdownPct = riskStatus?.current_daily_drawdown_pct ?? 0;
  const maxDrawdownPct = riskStatus?.max_daily_drawdown_pct ?? 5;

  return (
    <div className="space-y-6">
      {/* Risk Engine Status Banner */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-6 shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
              riskStatus?.circuit_breaker_tripped
                ? "bg-rose-500/10 border border-rose-500/30 text-rose-400"
                : "bg-emerald-500/10 border border-emerald-500/30 text-emerald-400"
            }`}>
              {riskStatus?.circuit_breaker_tripped ? (
                <ShieldAlert className="w-5 h-5" />
              ) : (
                <ShieldCheck className="w-5 h-5" />
              )}
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Centralized Risk Engine & Portfolio Safeguards
              </h2>
              <p className="text-xs text-slate-400">
                Independent pre-trade compliance checks enforced prior to any brokerage order transmission
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`px-3 py-1 rounded-full text-xs font-bold border ${
                riskStatus?.healthy
                  ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-400"
                  : "bg-rose-500/15 border-rose-500/40 text-rose-400"
              }`}
            >
              {riskStatus?.healthy ? "ALL LIMITS COMPLIANT" : "BREACH DETECTED"}
            </span>
          </div>
        </div>

        {/* Real-time Utilization Meters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-[#1E293B]">
          {/* Gross Exposure Utilization */}
          <div className="bg-[#1A2338]/60 border border-[#2D3A58]/50 rounded-lg p-4">
            <div className="flex justify-between text-xs font-mono mb-2">
              <span className="text-slate-400">Gross Capital Exposure</span>
              <span className="text-white font-bold">{exposurePct.toFixed(1)}% / {maxExposurePct.toFixed(0)}%</span>
            </div>
            <div className="w-full bg-[#131B2B] rounded-full h-2 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  exposurePct > maxExposurePct * 0.9 ? "bg-amber-500" : "bg-cyan-500"
                }`}
                style={{ width: `${Math.min(100, (exposurePct / maxExposurePct) * 100)}%` }}
              />
            </div>
            <span className="text-[10px] text-slate-500 mt-2 block">
              Max total capital invested across all holdings
            </span>
          </div>

          {/* Concurrent Positions Utilization */}
          <div className="bg-[#1A2338]/60 border border-[#2D3A58]/50 rounded-lg p-4">
            <div className="flex justify-between text-xs font-mono mb-2">
              <span className="text-slate-400">Concurrent Positions</span>
              <span className="text-white font-bold">{positionsCount} / {maxPositions} Slots</span>
            </div>
            <div className="w-full bg-[#131B2B] rounded-full h-2 overflow-hidden">
              <div
                className="h-full rounded-full bg-emerald-500 transition-all"
                style={{ width: `${Math.min(100, (positionsCount / maxPositions) * 100)}%` }}
              />
            </div>
            <span className="text-[10px] text-slate-500 mt-2 block">
              Max concurrent distinct open positions
            </span>
          </div>

          {/* Daily Drawdown Utilization */}
          <div className="bg-[#1A2338]/60 border border-[#2D3A58]/50 rounded-lg p-4">
            <div className="flex justify-between text-xs font-mono mb-2">
              <span className="text-slate-400">Daily Drawdown</span>
              <span className="text-white font-bold">{drawdownPct.toFixed(2)}% / {maxDrawdownPct.toFixed(0)}%</span>
            </div>
            <div className="w-full bg-[#131B2B] rounded-full h-2 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  drawdownPct > maxDrawdownPct * 0.7 ? "bg-rose-500" : "bg-cyan-500"
                }`}
                style={{ width: `${Math.min(100, (drawdownPct / maxDrawdownPct) * 100)}%` }}
              />
            </div>
            <span className="text-[10px] text-slate-500 mt-2 block">
              Circuit breaker trips if daily loss reaches threshold
            </span>
          </div>
        </div>
      </div>

      {/* Configured Quantitative Limits Grid */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 shadow-lg">
        <h3 className="text-sm font-bold text-white tracking-tight mb-4">
          Configured Risk Boundaries
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
          <div className="p-3 rounded-lg bg-[#1A2338]/60 border border-[#2D3A58]/50">
            <span className="text-slate-400 block text-[11px] mb-1">Max Position Size</span>
            <b className="text-white text-base">
              {((riskConfig?.max_position_size_pct || 0.10) * 100).toFixed(0)}% of Equity
            </b>
          </div>
          <div className="p-3 rounded-lg bg-[#1A2338]/60 border border-[#2D3A58]/50">
            <span className="text-slate-400 block text-[11px] mb-1">Daily Loss Limit</span>
            <b className="text-rose-400 text-base">
              {((riskConfig?.max_daily_loss_pct || 0.05) * 100).toFixed(0)}% Circuit Breaker
            </b>
          </div>
          <div className="p-3 rounded-lg bg-[#1A2338]/60 border border-[#2D3A58]/50">
            <span className="text-slate-400 block text-[11px] mb-1">Model Entry Threshold</span>
            <b className="text-emerald-400 text-base">
              P(UP) &ge; {((riskConfig?.entry_threshold || 0.40) * 100).toFixed(0)}%
            </b>
          </div>
          <div className="p-3 rounded-lg bg-[#1A2338]/60 border border-[#2D3A58]/50">
            <span className="text-slate-400 block text-[11px] mb-1">Slippage & Commission</span>
            <b className="text-cyan-400 text-base">
              {riskConfig?.slippage_bps || 5.0} bps / {riskConfig?.commission_bps || 5.0} bps
            </b>
          </div>
        </div>
      </div>

      {/* Risk Events & Violations Log */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
            <AlertOctagon className="w-4 h-4 text-rose-400" />
            <span>Risk Events & Blocked Order Audit ({riskEvents.length})</span>
          </h3>
          <span className="text-xs text-slate-400 font-mono">Real-time Gating</span>
        </div>

        {riskEvents.length === 0 ? (
          <div className="py-8 text-center text-slate-500 text-xs">
            No risk violations or blocked orders recorded. All evaluated operations compliant.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-[#1E293B] text-slate-400">
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Rule Name</th>
                  <th className="py-2.5 px-3">Severity</th>
                  <th className="py-2.5 px-3">Description</th>
                  <th className="py-2.5 px-3">Blocked Order ID</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E293B]/60 text-slate-300">
                {riskEvents.map((ev) => (
                  <tr key={ev.id} className="hover:bg-[#1A2338]/40">
                    <td className="py-2.5 px-3 text-slate-400">
                      {new Date(ev.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2.5 px-3 font-bold text-white">
                      {ev.rule_name}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        ev.severity === "CRITICAL"
                          ? "bg-rose-500/20 text-rose-400 border border-rose-500/50"
                          : "bg-amber-500/20 text-amber-400 border border-amber-500/50"
                      }`}>
                        {ev.severity}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {ev.description}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 truncate max-w-[140px]">
                      {ev.blocked_order_id || "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Append-only Administrative Audit Trail */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 shadow-lg">
        <h3 className="text-sm font-bold text-white tracking-tight mb-4 flex items-center gap-2">
          <History className="w-4 h-4 text-cyan-400" />
          <span>Operational & Security Audit Trail ({auditLogs.length})</span>
        </h3>

        {auditLogs.length === 0 ? (
          <div className="py-8 text-center text-slate-500 text-xs">
            No audit events recorded.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-[#1E293B] text-slate-400">
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Action</th>
                  <th className="py-2.5 px-3">Entity</th>
                  <th className="py-2.5 px-3">Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E293B]/60 text-slate-300">
                {auditLogs.map((log) => (
                  <tr key={log.id} className="hover:bg-[#1A2338]/40">
                    <td className="py-2.5 px-3 text-slate-400">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2.5 px-3 font-bold text-cyan-400">
                      {log.action}
                    </td>
                    <td className="py-2.5 px-3 text-white">
                      {log.entity_type} {log.entity_id ? `(#${log.entity_id})` : ""}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300 truncate max-w-xs">
                      {log.details_json || "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
