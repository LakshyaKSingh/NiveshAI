"use client";

import React, { useState } from "react";
import {
  ShieldAlert,
  Play,
  Pause,
  Square,
  RefreshCw,
  AlertOctagon,
  Zap,
  Activity,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { AgentStatus, RiskStatus } from "@/lib/api";

interface HeaderProps {
  agentStatus: AgentStatus | null;
  riskStatus: RiskStatus | null;
  isConnected: boolean;
  onRefresh: () => void;
  onStartAgent: () => void;
  onPauseAgent: () => void;
  onStopAgent: () => void;
  onTriggerEmergencyStop: () => void;
  isTriggeringCycle: boolean;
  onTriggerCycle: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  agentStatus,
  riskStatus,
  isConnected,
  onRefresh,
  onStartAgent,
  onPauseAgent,
  onStopAgent,
  onTriggerEmergencyStop,
  isTriggeringCycle,
  onTriggerCycle,
}) => {
  const [showConfirmStop, setShowConfirmStop] = useState(false);

  const stateColors = {
    RUNNING: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40",
    PAUSED: "bg-amber-500/20 text-amber-400 border-amber-500/40",
    STOPPED: "bg-slate-500/20 text-slate-400 border-slate-500/40",
    ERROR: "bg-rose-500/20 text-rose-400 border-rose-500/40",
  };

  const currentState = agentStatus?.state || "STOPPED";

  return (
    <header className="bg-[#0D1424] border-b border-[#1E293B] sticky top-0 z-40 px-4 lg:px-8 py-3.5">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand & Status Pills */}
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white leading-none">
                TradePilot
              </h1>
              <span className="text-[10px] text-cyan-400 font-mono tracking-wider uppercase font-semibold">
                Autonomous Execution
              </span>
            </div>
          </div>

          <div className="h-5 w-px bg-[#1E293B] hidden sm:block" />

          {/* Paper Trading Guard Badge */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-semibold tracking-wide uppercase">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
            Paper Trading
          </div>

          {/* Connection Status */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
              isConnected
                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                : "bg-rose-500/10 border-rose-500/30 text-rose-400"
            }`}
          >
            {isConnected ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Backend Connected</span>
              </>
            ) : (
              <>
                <XCircle className="w-3.5 h-3.5" />
                <span>API Offline</span>
              </>
            )}
          </div>

          {/* Agent State Badge */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${
              stateColors[currentState] || stateColors.STOPPED
            }`}
          >
            <Activity className="w-3.5 h-3.5 animate-pulse" />
            <span>Agent: {currentState}</span>
          </div>

          {/* Circuit Breaker Status */}
          {riskStatus?.circuit_breaker_tripped && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-rose-500/20 border border-rose-500/50 text-rose-400 text-xs font-bold animate-pulse">
              <AlertOctagon className="w-3.5 h-3.5" />
              <span>CIRCUIT BREAKER ACTIVE</span>
            </div>
          )}
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2.5 flex-wrap">
          {/* Refresh / Run Cycle Button */}
          <button
            onClick={onTriggerCycle}
            disabled={isTriggeringCycle || !isConnected}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-[#1A2338] hover:bg-[#232F4A] text-slate-200 border border-[#2D3A58] transition-all disabled:opacity-50"
            title="Execute immediate ML inference and trading cycle"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${
                isTriggeringCycle ? "animate-spin text-cyan-400" : ""
              }`}
            />
            <span>{isTriggeringCycle ? "Evaluating..." : "Run Cycle"}</span>
          </button>

          {/* Agent Lifecycle Toggles */}
          {currentState === "STOPPED" && (
            <button
              onClick={onStartAgent}
              disabled={!isConnected}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm shadow-emerald-600/30 transition-all disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Start</span>
            </button>
          )}

          {currentState === "RUNNING" && (
            <>
              <button
                onClick={onPauseAgent}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold bg-amber-600 hover:bg-amber-500 text-white transition-all"
              >
                <Pause className="w-3.5 h-3.5 fill-current" />
                <span>Pause</span>
              </button>
              <button
                onClick={onStopAgent}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold bg-slate-700 hover:bg-slate-600 text-white transition-all"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
                <span>Stop</span>
              </button>
            </>
          )}

          {currentState === "PAUSED" && (
            <>
              <button
                onClick={onStartAgent}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition-all"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Resume</span>
              </button>
              <button
                onClick={onStopAgent}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold bg-slate-700 hover:bg-slate-600 text-white transition-all"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
                <span>Stop</span>
              </button>
            </>
          )}

          {/* Emergency Kill Switch Button */}
          <button
            onClick={() => setShowConfirmStop(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-bold bg-rose-600 hover:bg-rose-500 text-white shadow-sm shadow-rose-600/40 border border-rose-500 transition-all"
            title="CRITICAL: Cancel all orders and stop trading immediately"
          >
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>Emergency Kill</span>
          </button>
        </div>
      </div>

      {/* Emergency Kill Switch Confirmation Modal */}
      {showConfirmStop && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-[#131B2B] border border-rose-500/60 rounded-xl p-6 max-w-md w-full shadow-2xl shadow-rose-950/50">
            <div className="flex items-center gap-3 text-rose-400 mb-3">
              <AlertOctagon className="w-7 h-7" />
              <h3 className="text-lg font-bold text-white">
                Confirm Emergency Kill Switch
              </h3>
            </div>
            <p className="text-sm text-slate-300 mb-4 leading-relaxed">
              This will immediately halt all strategy execution, trip the
              system circuit breaker, and attempt to cancel all open orders on
              the paper brokerage. Holdings will NOT be liquidated.
            </p>
            <div className="flex items-center justify-end gap-3">
              <button
                onClick={() => setShowConfirmStop(false)}
                className="px-4 py-2 rounded-lg text-xs font-medium text-slate-300 hover:bg-[#1E293B] border border-[#2D3A58]"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowConfirmStop(false);
                  onTriggerEmergencyStop();
                }}
                className="px-4 py-2 rounded-lg text-xs font-bold bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/30"
              >
                CONFIRM EMERGENCY STOP
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
};
