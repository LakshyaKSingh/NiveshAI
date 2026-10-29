"use client";

import React, { useState } from "react";
import {
  ActiveModelMetadata,
  PredictionItem,
  SignalItem,
  api,
} from "@/lib/api";
import { Cpu, RefreshCw, Layers, CheckCircle2, Info } from "lucide-react";

interface PredictionsViewProps {
  modelMetadata: ActiveModelMetadata | null;
  predictions: PredictionItem[];
  signals: SignalItem[];
  onRefreshData: () => void;
}

export const PredictionsView: React.FC<PredictionsViewProps> = ({
  modelMetadata,
  predictions,
  signals,
  onRefreshData,
}) => {
  const [isRunningInference, setIsRunningInference] = useState(false);

  const handleRunInference = async () => {
    setIsRunningInference(true);
    try {
      await api.runInference();
      onRefreshData();
    } catch (e) {
      console.error(e);
    } finally {
      setIsRunningInference(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Active Model Specification Card */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-6 shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Active Production Model: {modelMetadata?.algorithm || "RandomForest"}
              </h2>
              <span className="text-xs text-slate-400 font-mono">
                Artifact: {modelMetadata?.artifact_path || "best_model.joblib"}
              </span>
            </div>
          </div>

          <button
            onClick={handleRunInference}
            disabled={isRunningInference}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold bg-cyan-600 hover:bg-cyan-500 text-white shadow-md shadow-cyan-600/30 transition-all disabled:opacity-50"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${
                isRunningInference ? "animate-spin" : ""
              }`}
            />
            <span>{isRunningInference ? "Evaluating Pipeline..." : "Run Universe Inference"}</span>
          </button>
        </div>

        {/* Feature Schema & Metrics Badges */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-[#1E293B]">
          <div>
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-2 font-medium">
              Expected Features ({modelMetadata?.num_features || 8})
            </span>
            <div className="flex flex-wrap gap-1.5">
              {(modelMetadata?.feature_names || [
                "return_1d", "return_5d", "return_10d", "volatility_10d",
                "volume_change", "RSI_14", "MACD", "MACD_signal"
              ]).map((feat) => (

                <span
                  key={feat}
                  className="px-2 py-0.5 rounded bg-[#1A2338] border border-[#2D3A58] text-[10px] font-mono text-cyan-300"
                >
                  {feat}
                </span>
              ))}
            </div>
          </div>

          <div>
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-2 font-medium">
              Test Set Evaluation Metrics
            </span>
            <div className="space-y-1 font-mono text-xs">
              <div className="flex justify-between text-slate-300">
                <span>Accuracy:</span>
                <b className="text-emerald-400">
                  {((modelMetadata?.metrics?.accuracy || 0.42) * 100).toFixed(1)}%
                </b>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Macro F1-Score:</span>
                <b className="text-cyan-400">
                  {(modelMetadata?.metrics?.macro_f1 || 0.38).toFixed(3)}
                </b>
              </div>
            </div>
          </div>

          <div>
            <span className="text-[11px] text-slate-400 uppercase tracking-wider block mb-2 font-medium">
              Strategy Classification Thresholds
            </span>
            <p className="text-xs text-slate-300 leading-relaxed">
              Long Entry: <b className="text-emerald-400 font-mono">P(UP) &ge; 0.40</b> and P(UP) &gt; P(DOWN).
              <br />
              Long Exit: <b className="text-rose-400 font-mono">P(DOWN) &ge; 0.40</b> and P(DOWN) &gt; P(UP).
            </p>
          </div>
        </div>
      </div>

      {/* Disclaimers & Distinction Notice */}
      <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-3.5 flex items-start gap-3 text-xs text-cyan-200">
        <Info className="w-4 h-4 shrink-0 text-cyan-400 mt-0.5" />
        <div>
          <b>Operational Distinction:</b> Probabilities reflect empirical model classifications, not guaranteed returns. 
          A generated signal (e.g. BUY) produces an Order Intent, which must independently satisfy all 10 Centralized Risk Engine checks 
          before being submitted as a paper order to the brokerage.
        </div>
      </div>

      {/* Predictions Table */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-bold text-white tracking-tight">
            Latest Prediction Records ({predictions.length})
          </h3>
          <span className="text-xs text-slate-400 font-mono">
            Most Recent First
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-[#1E293B] text-slate-400">
                <th className="py-2.5 px-3">Timestamp</th>
                <th className="py-2.5 px-3">Symbol</th>
                <th className="py-2.5 px-3">Predicted Class</th>
                <th className="py-2.5 px-3">P(UP)</th>
                <th className="py-2.5 px-3">P(NEUTRAL)</th>
                <th className="py-2.5 px-3">P(DOWN)</th>
                <th className="py-2.5 px-3">Probability Distribution</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E293B]/60 text-slate-300">
              {predictions.map((p, idx) => {
                const classColor =
                  p.predicted_class === "UP"
                    ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/30"
                    : p.predicted_class === "DOWN"
                    ? "text-rose-400 bg-rose-500/10 border-rose-500/30"
                    : "text-slate-300 bg-slate-500/10 border-slate-500/30";

                return (
                  <tr key={idx} className="hover:bg-[#1A2338]/40">
                    <td className="py-2.5 px-3 text-slate-400">
                      {new Date(p.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2.5 px-3 font-bold text-white">
                      {p.symbol}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${classColor}`}>
                        {p.predicted_class}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">
                      {(p.prob_up * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {(p.prob_neutral * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 text-rose-400 font-bold">
                      {(p.prob_down * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 w-48">
                      <div className="flex h-2 rounded-full overflow-hidden bg-[#1A2338]">
                        <div
                          className="bg-emerald-500"
                          style={{ width: `${p.prob_up * 100}%` }}
                          title={`P(UP): ${(p.prob_up * 100).toFixed(1)}%`}
                        />
                        <div
                          className="bg-slate-500"
                          style={{ width: `${p.prob_neutral * 100}%` }}
                          title={`P(NEUTRAL): ${(p.prob_neutral * 100).toFixed(1)}%`}
                        />
                        <div
                          className="bg-rose-500"
                          style={{ width: `${p.prob_down * 100}%` }}
                          title={`P(DOWN): ${(p.prob_down * 100).toFixed(1)}%`}
                        />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Strategy Signals Table */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 shadow-lg">
        <h3 className="text-sm font-bold text-white tracking-tight mb-4">
          Generated Strategy Signals & Sizing Decisions
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-[#1E293B] text-slate-400">
                <th className="py-2.5 px-3">Symbol</th>
                <th className="py-2.5 px-3">Signal</th>
                <th className="py-2.5 px-3">Threshold</th>
                <th className="py-2.5 px-3">Signal Reasoning</th>
                <th className="py-2.5 px-3">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E293B]/60 text-slate-300">
              {signals.map((s, idx) => (
                <tr key={idx} className="hover:bg-[#1A2338]/40">
                  <td className="py-2.5 px-3 font-bold text-white">
                    {s.symbol}
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                        s.signal === "BUY"
                          ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-400"
                          : s.signal === "EXIT"
                          ? "bg-rose-500/15 border-rose-500/40 text-rose-400"
                          : "bg-slate-500/15 border-slate-500/40 text-slate-400"
                      }`}
                    >
                      {s.signal}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-400">
                    &ge; {(s.threshold * 100).toFixed(0)}%
                  </td>
                  <td className="py-2.5 px-3 text-slate-300">
                    {s.reasoning}
                  </td>
                  <td className="py-2.5 px-3 text-slate-400">
                    {new Date(s.timestamp).toLocaleTimeString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
