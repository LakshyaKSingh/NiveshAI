"use client";

import React, { useState } from "react";
import { OrderItem, FillItem, api } from "@/lib/api";
import { Send, CheckCircle2, Clock, XCircle, AlertTriangle, ShieldCheck } from "lucide-react";

interface OrdersViewProps {
  orders: OrderItem[];
  trades: FillItem[];
  symbols: string[];
  onRefreshData: () => void;
}

export const OrdersView: React.FC<OrdersViewProps> = ({
  orders,
  trades,
  symbols,
  onRefreshData,
}) => {
  const [activeTab, setActiveTab] = useState<"orders" | "trades">("orders");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");

  // Manual Order Form State
  const [symbol, setSymbol] = useState("AAPL");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [qty, setQty] = useState<number>(5);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [orderFeedback, setOrderFeedback] = useState<{ type: "success" | "error"; msg: string } | null>(null);

  const handleSubmitOrder = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setOrderFeedback(null);
    try {
      const created = await api.submitOrder({
        symbol,
        side,
        qty: Number(qty),
        order_type: "market",
      });
      setOrderFeedback({
        type: "success",
        msg: `Order submitted: ${side.toUpperCase()} ${qty} ${symbol} -> Status: ${created.status}`,
      });
      onRefreshData();
    } catch (err: any) {
      setOrderFeedback({
        type: "error",
        msg: err.message || "Failed to submit order.",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const filteredOrders = orders.filter((o) => {
    if (statusFilter === "ALL") return true;
    return o.status.toUpperCase() === statusFilter;
  });

  return (
    <div className="space-y-6">
      {/* Manual Paper Order Entry Panel */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-bold text-white tracking-tight">
              Validated Manual Paper Order Entry
            </h3>
          </div>
          <span className="text-[11px] text-amber-400 bg-amber-500/10 border border-amber-500/30 px-2 py-0.5 rounded font-mono">
            Paper Broker Sandbox
          </span>
        </div>

        <form onSubmit={handleSubmitOrder} className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3.5 items-end">
          <div>
            <label className="text-[11px] font-medium text-slate-400 block mb-1.5">
              Symbol
            </label>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="w-full bg-[#1A2338] border border-[#2D3A58] rounded-lg px-3 py-1.5 text-xs font-mono font-bold text-white focus:outline-none focus:border-cyan-500"
            >
              {symbols.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-[11px] font-medium text-slate-400 block mb-1.5">
              Action Side
            </label>
            <div className="flex rounded-lg overflow-hidden border border-[#2D3A58]">
              <button
                type="button"
                onClick={() => setSide("buy")}
                className={`flex-1 py-1.5 text-xs font-bold transition-all ${
                  side === "buy"
                    ? "bg-emerald-600 text-white"
                    : "bg-[#1A2338] text-slate-400 hover:text-white"
                }`}
              >
                BUY
              </button>
              <button
                type="button"
                onClick={() => setSide("sell")}
                className={`flex-1 py-1.5 text-xs font-bold transition-all ${
                  side === "sell"
                    ? "bg-rose-600 text-white"
                    : "bg-[#1A2338] text-slate-400 hover:text-white"
                }`}
              >
                SELL
              </button>
            </div>
          </div>

          <div>
            <label className="text-[11px] font-medium text-slate-400 block mb-1.5">
              Quantity (Shares)
            </label>
            <input
              type="number"
              min="1"
              step="1"
              value={qty}
              onChange={(e) => setQty(Math.max(1, parseInt(e.target.value) || 1))}
              className="w-full bg-[#1A2338] border border-[#2D3A58] rounded-lg px-3 py-1.5 text-xs font-mono font-bold text-white focus:outline-none focus:border-cyan-500"
            />
          </div>

          <div>
            <label className="text-[11px] font-medium text-slate-400 block mb-1.5">
              Order Type
            </label>
            <input
              type="text"
              readOnly
              value="MARKET (DAY)"
              className="w-full bg-[#1A2338]/50 border border-[#2D3A58] rounded-lg px-3 py-1.5 text-xs font-mono text-slate-400 cursor-not-allowed"
            />
          </div>

          <div>
            <button
              type="submit"
              disabled={isSubmitting}
              className={`w-full py-2 rounded-lg text-xs font-bold text-white flex items-center justify-center gap-1.5 transition-all shadow-md ${
                side === "buy"
                  ? "bg-emerald-600 hover:bg-emerald-500 shadow-emerald-600/30"
                  : "bg-rose-600 hover:bg-rose-500 shadow-rose-600/30"
              } disabled:opacity-50`}
            >
              <Send className="w-3.5 h-3.5" />
              <span>{isSubmitting ? "Validating..." : `Submit ${side.toUpperCase()}`}</span>
            </button>
          </div>
        </form>

        {orderFeedback && (
          <div
            className={`mt-3 p-3 rounded-lg text-xs font-mono flex items-center gap-2 ${
              orderFeedback.type === "success"
                ? "bg-emerald-500/15 border border-emerald-500/40 text-emerald-300"
                : "bg-rose-500/15 border border-rose-500/40 text-rose-300"
            }`}
          >
            {orderFeedback.type === "success" ? (
              <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
            ) : (
              <XCircle className="w-4 h-4 shrink-0 text-rose-400" />
            )}
            <span>{orderFeedback.msg}</span>
          </div>
        )}
      </div>

      {/* Tabs: Orders vs Trades */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex rounded-lg overflow-hidden border border-[#2D3A58] p-1 bg-[#131B2B]">
          <button
            onClick={() => setActiveTab("orders")}
            className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all ${
              activeTab === "orders"
                ? "bg-cyan-500 text-black shadow-sm"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Order Lifecycle ({orders.length})
          </button>
          <button
            onClick={() => setActiveTab("trades")}
            className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all ${
              activeTab === "trades"
                ? "bg-cyan-500 text-black shadow-sm"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Execution Fills ({trades.length})
          </button>
        </div>

        {activeTab === "orders" && (
          <div className="flex items-center gap-1.5 text-xs font-mono">
            {["ALL", "FILLED", "SUBMITTED", "REJECTED", "CANCELED"].map((st) => (
              <button
                key={st}
                onClick={() => setStatusFilter(st)}
                className={`px-2.5 py-1 rounded text-[11px] font-semibold border transition-all ${
                  statusFilter === st
                    ? "bg-[#232F4A] text-white border-cyan-500/60"
                    : "bg-[#1A2338] text-slate-400 border-[#2D3A58] hover:text-white"
                }`}
              >
                {st}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Active Tab Table Content */}
      <div className="bg-[#131B2B] border border-[#1E293B] rounded-xl p-5 shadow-lg">
        {activeTab === "orders" ? (
          filteredOrders.length === 0 ? (
            <div className="py-12 text-center text-slate-500 text-xs">
              No orders found matching filter criteria.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-[#1E293B] text-slate-400">
                    <th className="py-2.5 px-3">Client Order ID</th>
                    <th className="py-2.5 px-3">Symbol</th>
                    <th className="py-2.5 px-3">Side</th>
                    <th className="py-2.5 px-3 text-right">Quantity</th>
                    <th className="py-2.5 px-3 text-right">Avg Fill Price</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3">Submitted At</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1E293B]/60 text-slate-300">
                  {filteredOrders.map((ord) => {
                    const isBuy = ord.side.toLowerCase() === "buy";
                    const statusColor =
                      ord.status === "FILLED"
                        ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/30"
                        : ord.status === "SUBMITTED"
                        ? "text-cyan-400 bg-cyan-500/10 border-cyan-500/30"
                        : ord.status === "REJECTED"
                        ? "text-rose-400 bg-rose-500/10 border-rose-500/30"
                        : "text-slate-400 bg-slate-500/10 border-slate-500/30";

                    return (
                      <tr key={ord.id} className="hover:bg-[#1A2338]/40">
                        <td className="py-2.5 px-3 text-slate-400 truncate max-w-[140px]" title={ord.client_order_id}>
                          {ord.client_order_id}
                        </td>
                        <td className="py-2.5 px-3 font-bold text-white">
                          {ord.symbol}
                        </td>
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            isBuy ? "text-emerald-400 bg-emerald-500/15" : "text-rose-400 bg-rose-500/15"
                          }`}>
                            {ord.side.toUpperCase()}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right text-white font-bold">
                          {ord.qty}
                        </td>
                        <td className="py-2.5 px-3 text-right font-bold text-cyan-300">
                          {ord.avg_fill_price ? `$${ord.avg_fill_price.toFixed(2)}` : "-"}
                        </td>
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${statusColor}`}>
                            {ord.status}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-slate-400">
                          {new Date(ord.submitted_at).toLocaleTimeString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )
        ) : (
          trades.length === 0 ? (
            <div className="py-12 text-center text-slate-500 text-xs">
              No execution fills recorded yet.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-[#1E293B] text-slate-400">
                    <th className="py-2.5 px-3">Broker Order ID</th>
                    <th className="py-2.5 px-3">Symbol</th>
                    <th className="py-2.5 px-3">Side</th>
                    <th className="py-2.5 px-3 text-right">Shares Filled</th>
                    <th className="py-2.5 px-3 text-right">Execution Price</th>
                    <th className="py-2.5 px-3 text-right">Commission Fee</th>
                    <th className="py-2.5 px-3">Fill Timestamp</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1E293B]/60 text-slate-300">
                  {trades.map((tr) => (
                    <tr key={tr.id} className="hover:bg-[#1A2338]/40">
                      <td className="py-2.5 px-3 text-slate-400 truncate max-w-[140px]">
                        {tr.broker_order_id}
                      </td>
                      <td className="py-2.5 px-3 font-bold text-white">
                        {tr.symbol}
                      </td>
                      <td className="py-2.5 px-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          tr.side.toLowerCase() === "buy" ? "text-emerald-400 bg-emerald-500/15" : "text-rose-400 bg-rose-500/15"
                        }`}>
                          {tr.side.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-right text-white font-bold">
                        {tr.shares}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold text-cyan-300">
                        ${tr.price.toFixed(2)}
                      </td>
                      <td className="py-2.5 px-3 text-right text-slate-400">
                        ${tr.fee.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 text-slate-400">
                        {new Date(tr.timestamp).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        )}
      </div>
    </div>
  );
};
