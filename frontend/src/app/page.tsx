"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  api,
  PortfolioSummary,
  PositionItem,
  EquityPoint,
  MarketPrice,
  PredictionItem,
  SignalItem,
  ActiveModelMetadata,
  OrderItem,
  FillItem,
  AgentStatus,
  RiskStatus,
  RiskConfig,
  RiskEvent,
  AuditLog,
} from "@/lib/api";
import { Header } from "@/components/Header";
import { DashboardView } from "@/components/views/DashboardView";
import { MarketsView } from "@/components/views/MarketsView";
import { PredictionsView } from "@/components/views/PredictionsView";
import { PortfolioView } from "@/components/views/PortfolioView";
import { OrdersView } from "@/components/views/OrdersView";
import { AgentControlView } from "@/components/views/AgentControlView";
import { RiskView } from "@/components/views/RiskView";
import {
  LayoutDashboard,
  BarChart3,
  Cpu,
  Briefcase,
  Layers,
  Settings2,
  ShieldAlert,
  RefreshCw,
  AlertCircle,
} from "lucide-react";

export default function Home() {
  const [activeTab, setActiveTab] = useState<string>("dashboard");
  const [isConnected, setIsConnected] = useState<boolean>(true);
  const [isInitialLoading, setIsInitialLoading] = useState<boolean>(true);
  const [isTriggeringCycle, setIsTriggeringCycle] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Application Data States
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);
  const [positions, setPositions] = useState<PositionItem[]>([]);
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [latestPrices, setLatestPrices] = useState<MarketPrice[]>([]);
  const [predictions, setPredictions] = useState<PredictionItem[]>([]);
  const [signals, setSignals] = useState<SignalItem[]>([]);
  const [modelMetadata, setModelMetadata] = useState<ActiveModelMetadata | null>(null);
  const [orders, setOrders] = useState<OrderItem[]>([]);
  const [trades, setTrades] = useState<FillItem[]>([]);
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [riskStatus, setRiskStatus] = useState<RiskStatus | null>(null);
  const [riskConfig, setRiskConfig] = useState<RiskConfig | null>(null);
  const [riskEvents, setRiskEvents] = useState<RiskEvent[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);

  // Authenticate & Load Data
  const loadAllData = useCallback(async () => {
    try {
      // 1. Ensure token exists; auto-login with default admin if needed
      try {
        await api.getMe();
      } catch {
        await api.login("admin", "TradePilotAdmin2026!");
      }

      // 2. Fetch parallel endpoints
      const [
        portData,
        posData,
        eqData,
        symData,
        pricesData,
        predData,
        sigData,
        modelData,
        ordersData,
        tradesData,
        agentData,
        riskStatData,
        riskCfgData,
        riskEvData,
        auditData,
      ] = await Promise.allSettled([
        api.getPortfolioSummary(),
        api.getPositions(),
        api.getEquityHistory(100),
        api.getSymbols(),
        api.getLatestPrices(),
        api.getLatestPredictions(30),
        api.getSignals(30),
        api.getActiveModel(),
        api.getOrders(50),
        api.getTrades(50),
        api.getAgentStatus(),
        api.getRiskStatus(),
        api.getRiskConfig(),
        api.getRiskEvents(50),
        api.getAuditLogs(50),
      ]);

      if (portData.status === "fulfilled") setPortfolio(portData.value);
      if (posData.status === "fulfilled") setPositions(posData.value);
      if (eqData.status === "fulfilled") setEquityHistory(eqData.value.points);
      if (symData.status === "fulfilled") setSymbols(symData.value);
      if (pricesData.status === "fulfilled") setLatestPrices(pricesData.value.prices);
      if (predData.status === "fulfilled") setPredictions(predData.value.predictions);
      if (sigData.status === "fulfilled") setSignals(sigData.value.signals);
      if (modelData.status === "fulfilled") setModelMetadata(modelData.value);
      if (ordersData.status === "fulfilled") setOrders(ordersData.value.orders);
      if (tradesData.status === "fulfilled") setTrades(tradesData.value.fills);
      if (agentData.status === "fulfilled") setAgentStatus(agentData.value);
      if (riskStatData.status === "fulfilled") setRiskStatus(riskStatData.value);
      if (riskCfgData.status === "fulfilled") setRiskConfig(riskCfgData.value);
      if (riskEvData.status === "fulfilled") setRiskEvents(riskEvData.value.events);
      if (auditData.status === "fulfilled") setAuditLogs(auditData.value.logs);

      setIsConnected(true);
      setErrorMessage(null);
    } catch (err: any) {
      console.warn("Connection error:", err);
      setIsConnected(false);
      setErrorMessage("Unable to connect to TradePilot Backend API (http://localhost:8000). Ensure the backend server is running.");
    } finally {
      setIsInitialLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAllData();
    // Auto-refresh interval (10 seconds)
    const interval = setInterval(() => {
      loadAllData();
    }, 10000);
    return () => clearInterval(interval);
  }, [loadAllData]);

  // Agent Controls
  const handleStartAgent = async () => {
    try {
      await api.startAgent();
      await loadAllData();
    } catch (e: any) {
      alert(`Start Agent failed: ${e.message}`);
    }
  };

  const handlePauseAgent = async () => {
    try {
      await api.pauseAgent();
      await loadAllData();
    } catch (e: any) {
      alert(`Pause Agent failed: ${e.message}`);
    }
  };

  const handleStopAgent = async () => {
    try {
      await api.stopAgent();
      await loadAllData();
    } catch (e: any) {
      alert(`Stop Agent failed: ${e.message}`);
    }
  };

  const handleEmergencyStop = async () => {
    try {
      const res = await api.emergencyStop();
      alert(`Emergency Kill Switch Triggered! ${res.cancelled_orders_count} orders cancelled.`);
      await loadAllData();
    } catch (e: any) {
      alert(`Emergency Stop failed: ${e.message}`);
    }
  };

  const handleTriggerCycle = async () => {
    setIsTriggeringCycle(true);
    try {
      await api.runInference();
      await loadAllData();
    } catch (e: any) {
      alert(`Cycle evaluation failed: ${e.message}`);
    } finally {
      setIsTriggeringCycle(false);
    }
  };

  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "markets", label: "Markets", icon: BarChart3 },
    { id: "predictions", label: "Predictions & Signals", icon: Cpu },
    { id: "portfolio", label: "Portfolio", icon: Briefcase },
    { id: "orders", label: "Orders & Trades", icon: Layers },
    { id: "agent", label: "Agent Control", icon: Settings2 },
    { id: "risk", label: "Risk & Activity", icon: ShieldAlert },
  ];

  return (
    <div className="min-h-screen bg-[#080D1A] text-slate-100 flex flex-col font-sans">
      {/* Top Application Header */}
      <Header
        agentStatus={agentStatus}
        riskStatus={riskStatus}
        isConnected={isConnected}
        onRefresh={loadAllData}
        onStartAgent={handleStartAgent}
        onPauseAgent={handlePauseAgent}
        onStopAgent={handleStopAgent}
        onTriggerEmergencyStop={handleEmergencyStop}
        isTriggeringCycle={isTriggeringCycle}
        onTriggerCycle={handleTriggerCycle}
      />

      {/* Disconnected Alert Banner */}
      {!isConnected && (
        <div className="bg-rose-500/15 border-b border-rose-500/30 px-6 py-2.5 text-xs text-rose-300 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>
              {errorMessage || "Disconnected from TradePilot API server."}
            </span>
          </div>
          <button
            onClick={loadAllData}
            className="flex items-center gap-1 font-bold underline hover:text-white"
          >
            <RefreshCw className="w-3 h-3" />
            <span>Retry Connection</span>
          </button>
        </div>
      )}

      {/* Primary Navigation Tabs */}
      <nav className="bg-[#0B1020] border-b border-[#1E293B] sticky top-[57px] z-30 px-4 lg:px-8">
        <div className="max-w-7xl mx-auto flex items-center gap-1 overflow-x-auto py-2 scrollbar-none">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                  isActive
                    ? "bg-[#1A2338] text-cyan-400 border border-[#2D3A58] shadow-sm shadow-cyan-950/40"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#131B2B]"
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-cyan-400" : "text-slate-500"}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </nav>

      {/* Main Content View Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 lg:p-8">
        {isInitialLoading ? (
          <div className="h-96 flex flex-col items-center justify-center gap-3 text-slate-400">
            <RefreshCw className="w-8 h-8 animate-spin text-cyan-400" />
            <span className="text-sm font-mono">Synchronizing TradePilot State...</span>
          </div>
        ) : (
          <>
            {activeTab === "dashboard" && (
              <DashboardView
                portfolio={portfolio}
                positions={positions}
                equityHistory={equityHistory}
                predictions={predictions}
                signals={signals}
                riskStatus={riskStatus}
                onNavigateTab={setActiveTab}
              />
            )}

            {activeTab === "markets" && (
              <MarketsView
                symbols={symbols.length > 0 ? symbols : ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ"]}
                latestPrices={latestPrices}
              />
            )}

            {activeTab === "predictions" && (
              <PredictionsView
                modelMetadata={modelMetadata}
                predictions={predictions}
                signals={signals}
                onRefreshData={loadAllData}
              />
            )}

            {activeTab === "portfolio" && (
              <PortfolioView
                portfolio={portfolio}
                positions={positions}
              />
            )}

            {activeTab === "orders" && (
              <OrdersView
                orders={orders}
                trades={trades}
                symbols={symbols.length > 0 ? symbols : ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ"]}
                onRefreshData={loadAllData}
              />
            )}

            {activeTab === "agent" && (
              <AgentControlView
                agentStatus={agentStatus}
                riskStatus={riskStatus}
                onRefreshData={loadAllData}
                onStartAgent={handleStartAgent}
                onPauseAgent={handlePauseAgent}
                onStopAgent={handleStopAgent}
                onTriggerEmergencyStop={handleEmergencyStop}
              />
            )}

            {activeTab === "risk" && (
              <RiskView
                riskStatus={riskStatus}
                riskConfig={riskConfig}
                riskEvents={riskEvents}
                auditLogs={auditLogs}
              />
            )}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-[#1E293B] bg-[#090E1D] py-4 px-6 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>TradePilot &bull; ML-Driven Automated Trading System</span>
          <span className="text-amber-500/80">&bull; STRICTLY PAPER TRADING ONLY &bull;</span>
          <span>v1.0.0</span>
        </div>
      </footer>
    </div>
  );
}
