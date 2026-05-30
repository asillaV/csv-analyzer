import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { 
  Activity, Check, Play, Settings2, Info, FileSpreadsheet, ChevronDown 
} from 'lucide-react';
import { Card } from '../../ui/card';
import { Button } from '../../ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../ui/select';
import { Input } from '../../ui/input';
import { Label } from '../../ui/label';
import { Switch } from '../../ui/switch';
import { Badge } from '../../ui/badge';

// Generate mock data for the chart
const generateData = () => {
  return Array.from({ length: 100 }, (_, i) => {
    const x = i * 0.01;
    const baseSignal = Math.sin(x * 15) * 2;
    const noise = (Math.random() - 0.5) * 1.5;
    return {
      time_s: x.toFixed(2),
      original: baseSignal + noise,
      filtered: baseSignal
    };
  });
};

export function Workflow() {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    setData(generateData());
  }, []);

  return (
    <div className="min-h-screen bg-[#fafafa] font-sans text-slate-900 flex flex-col">
      <style dangerouslySetInnerHTML={{__html: `
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
          font-family: 'Inter', sans-serif;
        }
        
        @keyframes pulse-ring {
          0% { transform: scale(0.8); box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.7); }
          70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
          100% { transform: scale(0.8); box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
        }
        
        .step-active-ring {
          animation: pulse-ring 2s infinite;
        }
      `}} />

      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center gap-3 shrink-0">
        <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white">
          <Activity size={18} />
        </div>
        <h1 className="font-semibold text-lg text-slate-900">Analizzatore CSV</h1>
      </header>

      {/* Progress Bar */}
      <div className="bg-white border-b border-slate-200 px-8 py-5 shrink-0">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center justify-between relative">
            <div className="absolute left-0 top-1/2 -translate-y-1/2 w-full h-0.5 bg-slate-100 -z-10"></div>
            
            {/* Step 1 */}
            <div className="flex flex-col items-center gap-2 bg-white px-2">
              <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white shadow-sm">
                <Check size={16} strokeWidth={3} />
              </div>
              <span className="text-sm font-medium text-slate-600">① Carica file</span>
            </div>

            {/* Step 2 */}
            <div className="flex flex-col items-center gap-2 bg-white px-2">
              <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white shadow-sm">
                <Check size={16} strokeWidth={3} />
              </div>
              <span className="text-sm font-medium text-slate-600">② Configura</span>
            </div>

            {/* Step 3 */}
            <div className="flex flex-col items-center gap-2 bg-white px-2">
              <div className="relative">
                <div className="absolute inset-0 rounded-full step-active-ring"></div>
                <div className="w-8 h-8 rounded-full border-2 border-indigo-600 bg-white flex items-center justify-center text-indigo-600 font-bold shadow-sm relative z-10">
                  3
                </div>
              </div>
              <span className="text-sm font-semibold text-indigo-600">③ Analizza</span>
            </div>

            {/* Step 4 */}
            <div className="flex flex-col items-center gap-2 bg-white px-2">
              <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 font-bold shadow-sm">
                4
              </div>
              <span className="text-sm font-medium text-slate-400">④ Report</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8 flex flex-col md:flex-row gap-6 md:gap-8">
        
        {/* Left Panel - Controls */}
        <div className="w-full md:w-[340px] flex-shrink-0 flex flex-col gap-6">
          <Card className="p-5 border-slate-200 shadow-sm flex flex-col h-full bg-white rounded-xl">
            <h2 className="text-lg font-semibold text-indigo-600 mb-6 flex items-center gap-2">
              ③ Analizza
            </h2>

            <div className="space-y-6 flex-1">
              {/* Columns Section */}
              <div className="space-y-4">
                <h3 className="text-sm font-semibold text-slate-800 uppercase tracking-wider">Colonne</h3>
                
                <div className="space-y-3">
                  <div className="space-y-1.5">
                    <Label className="text-xs text-slate-500">Asse X (Tempo)</Label>
                    <Select defaultValue="time">
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Seleziona..." />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="time">time_s</SelectItem>
                        <SelectItem value="index">index</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1.5">
                    <Label className="text-xs text-slate-500">Asse Y (Segnale)</Label>
                    <Select defaultValue="accel">
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Seleziona..." />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="accel">acceleration_g</SelectItem>
                        <SelectItem value="temp">temperature_C</SelectItem>
                        <SelectItem value="vel">velocity_m_s</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <div className="h-px bg-slate-100"></div>

              {/* Sampling Section */}
              <div className="space-y-4">
                <h3 className="text-sm font-semibold text-slate-800 uppercase tracking-wider">Campionamento (fs)</h3>
                
                <div className="space-y-2">
                  <Input type="number" defaultValue="0" className="w-full" />
                  <p className="text-xs text-slate-500 ml-1">0 = automatico</p>
                  
                  <div className="mt-2 bg-blue-50 text-blue-700 px-3 py-2 rounded-md text-xs flex items-start gap-2 border border-blue-100">
                    <Info size={14} className="mt-0.5 shrink-0" />
                    <span>fs ≈ 1000 Hz (stimata da indice)</span>
                  </div>
                </div>
              </div>

              <div className="h-px bg-slate-100"></div>

              {/* Filter Section */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                    Filtro <Settings2 size={14} className="text-slate-400" />
                  </h3>
                  <Switch defaultChecked id="filter-enable" />
                </div>
                
                <div className="space-y-3 p-3 bg-slate-50 rounded-lg border border-slate-100">
                  <div className="space-y-1.5">
                    <Label className="text-xs text-slate-500">Tipo Filtro</Label>
                    <Select defaultValue="butter_lp">
                      <SelectTrigger className="w-full bg-white h-8 text-sm">
                        <SelectValue placeholder="Seleziona..." />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="butter_lp">Butterworth LP</SelectItem>
                        <SelectItem value="butter_hp">Butterworth HP</SelectItem>
                        <SelectItem value="butter_bp">Butterworth BP</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1.5">
                      <Label className="text-xs text-slate-500">Ordine</Label>
                      <Input type="number" defaultValue="4" className="h-8 text-sm bg-white" />
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-xs text-slate-500">Cutoff (Hz)</Label>
                      <Input type="number" defaultValue="5" className="h-8 text-sm bg-white" />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <Button className="w-full mt-6 bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm flex items-center gap-2 py-6">
              <Play size={16} fill="currentColor" /> Applica analisi
            </Button>
          </Card>
        </div>

        {/* Right Panel - Chart & Results */}
        <div className="flex-1 min-w-0 flex flex-col gap-4">
          <Card className="flex-1 bg-white border-slate-200 shadow-sm rounded-xl overflow-hidden flex flex-col">
            <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between bg-white">
              <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                <Activity size={18} className="text-indigo-500" />
                Risultato analisi — <span className="text-indigo-600 font-mono text-sm bg-indigo-50 px-2 py-1 rounded">acceleration_g</span>
              </h2>
            </div>
            
            <div className="flex-1 p-6 min-h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis 
                    dataKey="time_s" 
                    tick={{ fontSize: 12, fill: '#64748b' }}
                    tickLine={false}
                    axisLine={{ stroke: '#cbd5e1' }}
                    tickMargin={10}
                    minTickGap={30}
                  />
                  <YAxis 
                    tick={{ fontSize: 12, fill: '#64748b' }}
                    tickLine={false}
                    axisLine={false}
                    tickMargin={10}
                  />
                  <Tooltip 
                    contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                    labelStyle={{ fontWeight: 600, color: '#0f172a', marginBottom: '4px' }}
                  />
                  <Legend 
                    verticalAlign="top" 
                    height={36} 
                    iconType="circle"
                    wrapperStyle={{ fontSize: '13px', fontWeight: 500 }}
                  />
                  <Line 
                    name="Originale (rumore)" 
                    type="monotone" 
                    dataKey="original" 
                    stroke="#93c5fd" 
                    strokeWidth={1.5}
                    dot={false}
                    activeDot={{ r: 4, fill: '#3b82f6' }}
                    isAnimationActive={false}
                  />
                  <Line 
                    name="Filtrato (Butter LP 5Hz)" 
                    type="monotone" 
                    dataKey="filtered" 
                    stroke="#ef4444" 
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 4, fill: '#ef4444' }}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <div className="flex flex-wrap gap-3 mt-2">
            <div className="bg-emerald-50 text-emerald-700 border border-emerald-100 px-3 py-1.5 rounded-full text-sm font-medium flex items-center gap-1.5">
              <Check size={14} className="shrink-0" />
              <span>1 200 campioni</span>
            </div>
            <div className="bg-emerald-50 text-emerald-700 border border-emerald-100 px-3 py-1.5 rounded-full text-sm font-medium flex items-center gap-1.5">
              <Check size={14} className="shrink-0" />
              <span>Filtro applicato: Butter LP 5 Hz</span>
            </div>
            <div className="bg-emerald-50 text-emerald-700 border border-emerald-100 px-3 py-1.5 rounded-full text-sm font-medium flex items-center gap-1.5">
              <Check size={14} className="shrink-0" />
              <span>fs: 1000 Hz</span>
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}
