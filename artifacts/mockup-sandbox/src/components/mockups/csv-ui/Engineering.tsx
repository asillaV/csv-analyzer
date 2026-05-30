import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip } from 'recharts';
import { Download, Play, Settings, RefreshCcw, Activity, FileSpreadsheet, ChevronDown, Check, SlidersHorizontal, Power } from 'lucide-react';

const mockData = Array.from({ length: 150 }).map((_, i) => {
  const t = i * 0.001;
  const noise = (Math.random() - 0.5) * 0.8;
  const signal = Math.sin(t * Math.PI * 40) + Math.sin(t * Math.PI * 10) * 0.5;
  return {
    time: t.toFixed(3),
    original: Number((signal + noise).toFixed(3)),
    filtered: Number(signal.toFixed(3))
  };
});

export function Engineering() {
  const [filterEnabled, setFilterEnabled] = useState(true);
  const [fftEnabled, setFftEnabled] = useState(false);

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans text-sm selection:bg-emerald-900/50">
      <style dangerouslySetInnerHTML={{ __html: `
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');
        .font-mono { font-family: 'JetBrains Mono', monospace; }
        .font-sans { font-family: 'Inter', sans-serif; }
      `}} />
      
      {/* Top Bar */}
      <header className="h-12 bg-slate-900 border-b border-slate-800 flex items-center justify-between px-4 shrink-0">
        <div className="flex items-center gap-4">
          <div className="font-mono text-emerald-500 font-bold tracking-tight">
            [ CSV_ANALYZER ]
          </div>
          <div className="w-px h-4 bg-slate-700" />
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)] animate-pulse" />
            <span className="text-emerald-500">CONNECTED</span>
          </div>
          <div className="w-px h-4 bg-slate-700" />
          <div className="flex items-center gap-2 text-slate-400 font-mono text-xs">
            <FileSpreadsheet size={14} />
            vibration_data.csv
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded transition-colors text-xs font-mono">
            <Download size={14} />
            EXPORT
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <aside className="w-[260px] bg-slate-900 border-r border-slate-800 flex flex-col shrink-0 overflow-y-auto custom-scrollbar">
          <div className="p-4 space-y-6">
            
            {/* SAMPLING SECTION */}
            <section className="space-y-3">
              <h3 className="text-[10px] font-bold text-emerald-500 tracking-widest uppercase flex items-center gap-2">
                <Activity size={12} /> Sampling
              </h3>
              
              <div className="space-y-2">
                <label className="text-xs text-slate-400 font-mono">FREQUENCY (fs)</label>
                <div className="relative">
                  <input 
                    type="text" 
                    defaultValue="1000" 
                    className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-slate-200 font-mono text-sm focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                  <div className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 font-mono text-xs">
                    Hz
                  </div>
                </div>
                <div className="inline-flex items-center gap-1.5 px-2 py-1 bg-emerald-950/30 border border-emerald-900/50 rounded text-emerald-400 text-[10px] font-mono mt-1">
                  <Check size={10} /> fs = 1000 Hz / manual
                </div>
              </div>
            </section>

            <div className="h-px bg-slate-800" />

            {/* FILTER SECTION */}
            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-[10px] font-bold text-emerald-500 tracking-widest uppercase flex items-center gap-2">
                  <SlidersHorizontal size={12} /> Filter
                </h3>
                <button 
                  onClick={() => setFilterEnabled(!filterEnabled)}
                  className={`w-8 h-4 rounded-full relative transition-colors ${filterEnabled ? 'bg-emerald-500' : 'bg-slate-700'}`}
                >
                  <div className={`w-3 h-3 rounded-full bg-white absolute top-0.5 transition-all ${filterEnabled ? 'left-4.5 right-0.5' : 'left-0.5'}`} style={{ left: filterEnabled ? '18px' : '2px' }} />
                </button>
              </div>

              <div className={`space-y-4 ${!filterEnabled && 'opacity-50 pointer-events-none'}`}>
                <div className="space-y-2">
                  <label className="text-xs text-slate-400 font-mono">TYPE</label>
                  <div className="relative">
                    <select className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-slate-200 font-mono text-sm appearance-none focus:outline-none focus:border-emerald-500 transition-colors">
                      <option>Butterworth LP</option>
                      <option>Butterworth HP</option>
                      <option>Butterworth BP</option>
                    </select>
                    <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-xs text-slate-400 font-mono">ORDER</label>
                  <div className="flex items-center gap-2">
                    <input type="range" min="1" max="10" defaultValue="4" className="flex-1 accent-emerald-500 h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer" />
                    <span className="text-sm font-mono w-4 text-right">4</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-2">
                    <label className="text-[10px] text-slate-500 font-mono">CUTOFF LO</label>
                    <input type="text" defaultValue="5" className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-200 font-mono text-sm focus:outline-none focus:border-emerald-500" />
                  </div>
                  <div className="space-y-2">
                    <label className="text-[10px] text-slate-500 font-mono">CUTOFF HI</label>
                    <input type="text" placeholder="-" className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-slate-400 font-mono text-sm focus:outline-none focus:border-emerald-500" />
                  </div>
                </div>
              </div>
            </section>

            <div className="h-px bg-slate-800" />

            {/* FFT SECTION */}
            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-[10px] font-bold text-emerald-500 tracking-widest uppercase flex items-center gap-2">
                  <Activity size={12} /> FFT
                </h3>
                <button 
                  onClick={() => setFftEnabled(!fftEnabled)}
                  className={`w-8 h-4 rounded-full relative transition-colors ${fftEnabled ? 'bg-emerald-500' : 'bg-slate-700'}`}
                >
                  <div className={`w-3 h-3 rounded-full bg-white absolute top-0.5 transition-all`} style={{ left: fftEnabled ? '18px' : '2px' }} />
                </button>
              </div>

              <div className={`space-y-3 ${!fftEnabled && 'opacity-50 pointer-events-none'}`}>
                <label className="flex items-center gap-2 cursor-pointer group">
                  <div className="w-4 h-4 rounded-full border border-slate-600 group-hover:border-emerald-500 flex items-center justify-center">
                    <div className="w-2 h-2 rounded-full bg-transparent" />
                  </div>
                  <span className="text-sm font-mono text-slate-300">Original signal</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer group">
                  <div className="w-4 h-4 rounded-full border border-emerald-500 flex items-center justify-center">
                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                  </div>
                  <span className="text-sm font-mono text-emerald-400">Filtered signal</span>
                </label>
              </div>
            </section>
          </div>
        </aside>

        {/* Main Area */}
        <main className="flex-1 flex flex-col min-w-0 bg-[#020617] p-4 gap-4">
          
          {/* Top Controls */}
          <div className="flex items-center justify-between bg-slate-900 border border-slate-800 rounded p-2 shadow-sm">
            <div className="flex items-center gap-4 px-2">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-slate-400">X:</span>
                <div className="relative">
                  <select className="bg-slate-950 border border-slate-700 rounded px-2 py-1 pr-8 text-slate-200 font-mono text-xs appearance-none focus:outline-none focus:border-emerald-500">
                    <option>time_s</option>
                    <option>index</option>
                  </select>
                  <ChevronDown size={12} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-slate-400">Y:</span>
                <div className="relative">
                  <select className="bg-slate-950 border border-slate-700 rounded px-2 py-1 pr-8 text-emerald-400 font-mono text-xs appearance-none focus:outline-none focus:border-emerald-500">
                    <option>acceleration_g</option>
                    <option>temperature_C</option>
                  </select>
                  <ChevronDown size={12} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
                </div>
              </div>
            </div>

            <button className="flex items-center gap-2 px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded transition-colors text-xs font-mono font-bold shadow-[0_0_10px_rgba(16,185,129,0.2)]">
              <Play size={14} className="fill-current" />
              APPLY / PLOT
            </button>
          </div>

          {/* Chart Area */}
          <div className="flex-1 bg-slate-900 border border-slate-800 rounded flex flex-col overflow-hidden relative shadow-lg">
            
            <div className="px-4 py-3 border-b border-slate-800 flex justify-between items-center bg-slate-900/50 backdrop-blur z-10">
              <h2 className="text-sm font-mono text-slate-200 font-medium">
                acceleration_g <span className="text-slate-500">vs</span> time_s
              </h2>
              <div className="flex gap-4 text-[10px] font-mono">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-500/50 border border-emerald-500" />
                  <span className="text-slate-400">Original</span>
                </div>
                {filterEnabled && (
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-amber-500" />
                    <span className="text-slate-300">Filtered</span>
                  </div>
                )}
              </div>
            </div>

            <div className="flex-1 p-2 min-h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis 
                    dataKey="time" 
                    stroke="#475569" 
                    tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }}
                    tickMargin={10}
                    minTickGap={30}
                  />
                  <YAxis 
                    stroke="#475569" 
                    tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }}
                    domain={['auto', 'auto']}
                    width={60}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '4px', padding: '8px' }}
                    itemStyle={{ fontFamily: 'monospace', fontSize: '12px' }}
                    labelStyle={{ fontFamily: 'monospace', fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="original" 
                    stroke="#10b981" 
                    strokeWidth={1} 
                    dot={false}
                    isAnimationActive={false}
                    opacity={filterEnabled ? 0.3 : 1}
                  />
                  {filterEnabled && (
                    <Line 
                      type="monotone" 
                      dataKey="filtered" 
                      stroke="#f59e0b" 
                      strokeWidth={2} 
                      dot={false}
                      isAnimationActive={false}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            {/* Grid overlay styling hint */}
            <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(30,41,59,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(30,41,59,0.1)_1px,transparent_1px)] bg-[size:20px_20px]" />
          </div>

          {/* Status Footer */}
          <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500 px-2 py-1">
            <span className="text-emerald-500/70">150 samples</span>
            <span>·</span>
            <span>Δt 0.001 s</span>
            {filterEnabled && (
              <>
                <span>·</span>
                <span className="text-amber-500/70">filter: butter_lp order 4</span>
                <span>·</span>
                <span>cutoff 5 Hz</span>
              </>
            )}
          </div>
          
        </main>
      </div>
    </div>
  );
}
