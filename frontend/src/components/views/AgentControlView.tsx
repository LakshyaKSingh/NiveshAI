"use client";

import React, { useState } from "react";
import { AgentStatus, RiskStatus, api } from "@/lib/api";
import {
  Play,
  Pause,
  Square,
  ShieldAlert,
  Activity,
  CheckCircle,
  AlertTriangle,
  Clock,
  Cpu,
  Layers,
} from "lucide-react";

interface AgentControlViewProps {
  agentStatus: AgentStatus | null;
  riskStatus: RiskStatus | null;
  onRefreshData: () => void;
  onStartAgent: () => void;
  onPauseAgent: () => void;
  onStopAgent: () => void;
  onTriggerEmergencyStop: () => void;
}

export const AgentControlView: React.FC<AgentControlViewProps> = ({
  agentStatus,
  riskStatus,
  onRefreshData,
  onStartAgent,
  onPauseAgent,
  onStopAgent,
  onTriggerEmergencyStop,
}) => {
  const currentState = agentStatus?.state || "STOPPED";

  return (
    <div className="space-y-6">
      {/* Primary Lifecycle Card */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-6 shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 pb-6 border-b border-[#1E293B]">
          <div>
            <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider font-mono">
              Agent Lifecycle Management
            </span>
            <h2 className="text-2xl font-bold text-white tracking-tight mt-1">
              Autonomous Strategy Engine
            </h2>
            <p className="text-xs text-slate-400 mt-1 max-w-xl leading-relaxed">
              Coordinates market data lookbacks, technical feature extraction, ML probability inference, 
              signal gating, and automated paper execution across all universe components.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {currentState === "STOPPED" && (
              <button
                onClick={onStartAgent}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/30 transition-all"
              >
                <Play className="w-4 h-4 fill-current" />
                <span>START TRADING AGENT</span>
              </button>
            )}

            {currentState === "RUNNING" && (
              <>
                <button
                  onClick={onPauseAgent}
                  className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-xs font-bold bg-amber-600 hover:bg-amber-500 text-white transition-all"
                >
                  <Pause className="w-4 h-4 fill-current" />
                  <span>PAUSE AGENT</span>
                </button>
                <button
                  onClick={onStopAgent}
                  className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-xs font-bold bg-slate-700 hover:bg-slate-600 text-white transition-all"
                >
                  <Square className="w-4 h-4 fill-current" />
                  <span>STOP AGENT</span>
                </button>
              </>
            )}

            {currentState === "PAUSED" && (
              <>
                <button
                  onClick={onStartAgent}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white transition-all"
                >
                  <Play className="w-4 h-4 fill-current" />
                  <span>RESUME AGENT</span>
                </button>
                <button
                  onClick={onStopAgent}
                  className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-xs font-bold bg-slate-700 hover:bg-slate-600 text-white transition-all"
                >
                  <Square className="w-4 h-4 fill-current" />
                  <span>STOP AGENT</span>
                </button>
              </>
            )}
          </div>
        </div>

        {/* State Indicators Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-6">
          <div className="bg-[#1A2338]/60 border border-[#2D3A58]/50 rounded-lg p-4">
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">
              Operational State
            </span>
            <div className="flex items-center gap-2 text-lg font-bold font-mono text-white">
              <span
                className={`w-3 h-3 rounded-full ${
                  currentState === "RUNNING"
                    ? "bg-emerald-400 animate-pulse"
                    : currentState === "PAUSED"
                    ? "bg-amber-400"
                    : "bg-slate-400"
                }`}
              />
              <span>{currentState}</span>
            </div>
            <span className="text-[10px] text-slate-500 mt-1 block">
              {currentState === "RUNNING"
                ? "Active scheduled order evaluation"
                : currentState === "PAUSED"
                ? "Monitors holdings; no new orders"
                : "Strategy loop is stopped"}
            </span>
          </div>

          <div className="bg-[#1A2338]/60 border border-[#2D3A58]/50 rounded-lg p-4">
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">
              Execution Mode
            </span>
            <div className="text-lg font-bold font-mono text-amber-400">
              PAPER TRADING
            </div>
            <span className="text-[10px] text-emerald-400 mt-1 block">
              &bull; Strict Paper Safeguard Active
            </span>
          </div>

          <div className="bg-[#1A2338]/60 border border-[#2D3A58]/50 rounded-lg p-4">
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">
              Brokerage Adapter
            </span>
            <div className="text-lg font-bold font-mono text-cyan-400">
              {agentStatus?.broker_type === "alpaca_paper" ? "Alpaca Paper" : "Mock Paper Engine"}
            </div>
            <span className="text-[10px] text-slate-400 mt-1 block">
              Slippage & Commission 5 bps
            </span>
          </div>

          <div className="bg-[#1A2338]/60 border border-[#2D3A58]/50 rounded-lg p-4">
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-1">
              Last Heartbeat
            </span>
            <div className="text-base font-bold font-mono text-white truncate">
              {agentStatus?.last_heartbeat
                ? new Date(agentStatus.last_heartbeat).toLocaleTimeString()
                : "Standing by"}
            </div>
            <span className="text-[10px] text-slate-400 mt-1 block">
              Worker interval: 60s
            </span>
          </div>
        </div>
      </div>

      {/* Emergency Kill Switch Section */}
      <div className="bg-rose-950/20 border border-rose-500/40 rounded-xl p-6 shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div>
            <div className="flex items-center gap-2.5 text-rose-400 mb-1">
              <ShieldAlert className="w-5 h-5" />
              <h3 className="text-base font-bold text-white tracking-tight">
                Emergency Stop & Kill-Switch
              </h3>
            </div>
            <p className="text-xs text-rose-200/80 max-w-xl leading-relaxed">
              Immediately trips the system circuit breaker, transitions agent to STOPPED state, 
              and executes broker cancel_all_orders to cancel any pending market or limit orders. 
              Holdings are preserved.
            </p>
          </div>

          <button
            onClick={onTriggerEmergencyStop}
            className="px-5 py-2.5 rounded-lg text-xs font-bold bg-rose-600 hover:bg-rose-500 text-white shadow-xl shadow-rose-950/60 border border-rose-400 transition-all shrink-0"
          >
            ACTIVATE EMERGENCY KILL
          </button>
        </div>
      </div>
    </div>
  );
};
