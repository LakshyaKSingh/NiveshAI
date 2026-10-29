"use client";

import React, { useState, useEffect } from "react";
import { BarItem, MarketPrice, api } from "@/lib/api";
import { PriceChart } from "../PriceChart";
import { Search, Activity, CheckCircle, RefreshCw } from "lucide-react";

interface MarketsViewProps {
  symbols: string[];
  latestPrices: MarketPrice[];
}

export const MarketsView: React.FC<MarketsViewProps> = ({
  symbols,
  latestPrices,
}) => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>("AAPL");
  const [bars, setBars] = useState<BarItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetchBars(selectedSymbol);
  }, [selectedSymbol]);

  const fetchBars = async (sym: string) => {
    setIsLoading(true);
    try {
      const data = await api.getHistoricalBars(sym, 40);
      setBars(data.bars);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  const filteredSymbols = symbols.filter((s) =>
    s.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const currentPriceObj = latestPrices.find(
    (p) => p.symbol.toUpperCase() === selectedSymbol.toUpperCase()
  );

  return (
    <div className="space-y-6">
      {/* Top Symbol Selector & Search */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-4 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 overflow-x-auto w-full md:w-auto pb-1 md:pb-0">
          {filteredSymbols.map((sym) => {
            const isSelected = sym === selectedSymbol;
            return (
              <button
                key={sym}
                onClick={() => setSelectedSymbol(sym)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-bold transition-all shrink-0 ${
                  isSelected
                    ? "bg-cyan-500 text-black shadow-md shadow-cyan-500/20"
                    : "bg-[#1A2338] hover:bg-[#232F4A] text-slate-300 border border-[#2D3A58]"
                }`}
              >
                {sym}
              </button>
            );
          })}
        </div>

        <div className="relative w-full md:w-56 shrink-0">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-3" />
          <input
            type="text"
            placeholder="Filter universe..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-[#1A2338] border border-[#2D3A58] rounded-lg pl-8 pr-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>
      </div>

      {/* Selected Stock Overview & Interactive Chart */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-6 shadow-lg">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-bold font-mono text-white tracking-tight">
                {selectedSymbol}
              </h2>
              <span className="text-xs px-2.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 font-semibold border border-cyan-500/30">
                Universe Component
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Alpaca Paper Data Feed &bull; 40-Day Historical Lookback
            </p>
          </div>

          {currentPriceObj && (
            <div className="text-left sm:text-right">
              <div className="text-2xl font-bold font-mono text-white">
                ${currentPriceObj.price.toFixed(2)}
              </div>
              <div
                className={`text-xs font-semibold ${
                  currentPriceObj.change_pct >= 0
                    ? "text-emerald-400"
                    : "text-rose-400"
                }`}
              >
                {currentPriceObj.change_pct >= 0 ? "+" : ""}
                {currentPriceObj.change_pct.toFixed(2)}% Today
              </div>
            </div>
          )}
        </div>

        {/* Chart View */}
        {isLoading ? (
          <div className="h-72 flex items-center justify-center text-slate-400 text-xs">
            <RefreshCw className="w-5 h-5 animate-spin mr-2 text-cyan-400" />
            Loading historical OHLCV data...
          </div>
        ) : (
          <PriceChart symbol={selectedSymbol} bars={bars} />
        )}
      </div>

      {/* Historical Bars Table */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-bold text-white tracking-tight">
            Recent Daily OHLCV Observations
          </h3>
          <span className="text-xs text-slate-400 font-mono">
            {bars.length} Bars Loaded
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-[#1E293B] text-slate-400">
                <th className="py-2.5 px-3">Date</th>
                <th className="py-2.5 px-3 text-right">Open</th>
                <th className="py-2.5 px-3 text-right">High</th>
                <th className="py-2.5 px-3 text-right">Low</th>
                <th className="py-2.5 px-3 text-right">Close</th>
                <th className="py-2.5 px-3 text-right">Volume</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E293B]/60 text-slate-300">
              {bars.slice(-10).reverse().map((bar, i) => (
                <tr key={i} className="hover:bg-[#1A2338]/40">
                  <td className="py-2.5 px-3 text-slate-400">
                    {new Date(bar.timestamp).toLocaleDateString()}
                  </td>
                  <td className="py-2.5 px-3 text-right">${bar.open.toFixed(2)}</td>
                  <td className="py-2.5 px-3 text-right">${bar.high.toFixed(2)}</td>
                  <td className="py-2.5 px-3 text-right">${bar.low.toFixed(2)}</td>
                  <td className="py-2.5 px-3 text-right font-bold text-white">
                    ${bar.close.toFixed(2)}
                  </td>
                  <td className="py-2.5 px-3 text-right text-slate-400">
                    {bar.volume.toLocaleString()}
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
