import React, { useState } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer
} from 'recharts';
import { 
  Settings, SlidersHorizontal, Activity, ChevronDown, ChevronRight, 
  Download, FileText, Bell, User, LayoutDashboard, Database
} from 'lucide-react';
import { Button } from '../../ui/button';
import { Input } from '../../ui/input';
import { Label } from '../../ui/label';
import { Switch } from '../../ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../ui/select';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../../ui/card';
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbSeparator, BreadcrumbPage } from '../../ui/breadcrumb';

// Generate mock signal data
const generateData = () => {
  const data = [];
  for (let i = 0; i < 200; i++) {
    const t = i / 1000;
    // Original signal: low freq + high freq noise
    const original = Math.sin(2 * Math.PI * 5 * t) + 0.5 * Math.sin(2 * Math.PI * 50 * t) + (Math.random() * 0.2 - 0.1);
    // Filtered signal: mostly low freq
    const filtered = Math.sin(2 * Math.PI * 5 * t) * 0.95;
    data.push({
      time: t.toFixed(3),
      original: original,
      filtered: filtered
    });
  }
  return data;
};

const mockData = generateData();

export function SaaS() {
  const [filterExpanded, setFilterExpanded] = useState(true);

  return (
    <div className="flex flex-col h-[100dvh] bg-slate-50 font-sans text-slate-900 overflow-hidden">
      <style dangerouslySetInnerHTML={{__html: `
        :root {
          --primary: 221 83% 53%; /* #2563eb */
        }
      `}} />

      {/* Top Navigation */}
      <header className="flex items-center justify-between h-14 px-4 bg-white border-b border-slate-200 shrink-0 z-10">
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-8 h-8 rounded bg-blue-600 text-white shadow-sm">
            <Activity size={18} strokeWidth={2.5} />
          </div>
          <span className="font-semibold text-slate-900 tracking-tight">CSV Analyzer</span>
          <div className="h-4 w-px bg-slate-200 mx-2"></div>
          <span className="text-sm font-medium text-slate-500">Workspace</span>
        </div>
        
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" className="text-slate-500 hover:text-slate-900">
            <Bell size={18} />
          </Button>
          <div className="flex items-center gap-2 pl-2 border-l border-slate-200">
            <div className="w-8 h-8 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-600 cursor-pointer hover:bg-slate-200 transition-colors">
              <User size={16} />
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <aside className="w-[270px] bg-slate-50/50 border-r border-slate-200 flex flex-col overflow-y-auto shrink-0 shadow-sm z-0">
          <div className="p-5 space-y-8">
            
            {/* Campionamento Section */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                <Settings size={16} className="text-slate-400" />
                <span>Campionamento</span>
              </div>
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="fs" className="text-xs text-slate-500 font-medium">Frequenza (Hz)</Label>
                  <Input id="fs" defaultValue="1000" className="h-9 bg-white border-slate-200 shadow-sm" />
                </div>
                <div className="bg-blue-50/80 border border-blue-100/80 rounded-lg p-3 flex items-start gap-2.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5 shrink-0 shadow-sm"></div>
                  <p className="text-xs text-blue-900 font-medium leading-relaxed">
                    fs ≈ 1000 Hz<br/>
                    <span className="text-blue-600/70 font-normal mt-0.5 inline-block">(da indice)</span>
                  </p>
                </div>
              </div>
            </div>

            <div className="h-px bg-slate-200/60"></div>

            {/* Filtro Section */}
            <div className="space-y-4">
              <div 
                className="flex items-center justify-between cursor-pointer group"
                onClick={() => setFilterExpanded(!filterExpanded)}
              >
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <SlidersHorizontal size={16} className="text-slate-400" />
                  <span>Filtro</span>
                </div>
                {filterExpanded ? <ChevronDown size={16} className="text-slate-400 group-hover:text-slate-600 transition-colors" /> : <ChevronRight size={16} className="text-slate-400 group-hover:text-slate-600 transition-colors" />}
              </div>

              {filterExpanded && (
                <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-5 shadow-sm">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium cursor-pointer" htmlFor="enable-filter">Abilita Filtro</Label>
                    <Switch id="enable-filter" defaultChecked />
                  </div>
                  
                  <div className="space-y-2">
                    <Label className="text-xs text-slate-500 font-medium">Tipo di Filtro</Label>
                    <Select defaultValue="lowpass">
                      <SelectTrigger className="h-9 text-sm border-slate-200">
                        <SelectValue placeholder="Seleziona..." />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="lowpass">Low-pass Butterworth</SelectItem>
                        <SelectItem value="highpass">High-pass Butterworth</SelectItem>
                        <SelectItem value="bandpass">Band-pass Butterworth</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-2">
                      <Label className="text-xs text-slate-500 font-medium">Ordine (N)</Label>
                      <Input type="number" defaultValue="4" className="h-9 border-slate-200" />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs text-slate-500 font-medium">Cutoff (Hz)</Label>
                      <Input type="number" defaultValue="10" className="h-9 border-slate-200" />
                    </div>
                  </div>
                </div>
              )}
            </div>

          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col bg-[#f8fafc] overflow-y-auto relative">
          <div className="p-8 max-w-6xl mx-auto w-full space-y-8 pb-20">
            
            {/* Header / Breadcrumbs */}
            <div className="flex items-center justify-between">
              <Breadcrumb>
                <BreadcrumbList>
                  <BreadcrumbItem>
                    <BreadcrumbLink href="#" className="flex items-center gap-1.5 text-slate-500 hover:text-slate-900 transition-colors">
                      <Database size={14} />
                      Dati
                    </BreadcrumbLink>
                  </BreadcrumbItem>
                  <BreadcrumbSeparator />
                  <BreadcrumbItem>
                    <BreadcrumbPage className="flex items-center gap-1.5 font-medium text-slate-900">
                      <FileText size={14} className="text-blue-600" />
                      vibration_data.csv
                    </BreadcrumbPage>
                  </BreadcrumbItem>
                </BreadcrumbList>
              </Breadcrumb>
              
              <div className="flex items-center gap-3">
                <Button variant="outline" size="sm" className="h-9 text-sm font-medium bg-white border-slate-200 shadow-sm">
                  <Download size={14} className="mr-2 text-slate-500" />
                  Esporta
                </Button>
              </div>
            </div>

            {/* Stats Chips */}
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-white border border-slate-200 rounded-lg shadow-sm text-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-300"></span>
                <span className="font-medium text-slate-700">1 200 righe</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 bg-white border border-slate-200 rounded-lg shadow-sm text-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-300"></span>
                <span className="font-medium text-slate-700">5 colonne</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 bg-white border border-slate-200 rounded-lg shadow-sm text-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
                <span className="font-medium text-slate-700">fs 1 kHz</span>
              </div>
            </div>

            {/* Controls Row */}
            <div className="grid grid-cols-1 md:grid-cols-[1fr_1fr_auto] gap-5 items-end bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
              <div className="space-y-2">
                <Label className="text-sm font-semibold text-slate-700">Asse X</Label>
                <Select defaultValue="time_s">
                  <SelectTrigger className="bg-slate-50 border-slate-200 h-10">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="time_s">time_s</SelectItem>
                    <SelectItem value="index">Index</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-sm font-semibold text-slate-700">Asse Y</Label>
                <Select defaultValue="acceleration_g">
                  <SelectTrigger className="bg-slate-50 border-slate-200 h-10">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="acceleration_g">acceleration_g</SelectItem>
                    <SelectItem value="temperature_C">temperature_C</SelectItem>
                    <SelectItem value="voltage_V">voltage_V</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button className="h-10 px-8 bg-blue-600 hover:bg-blue-700 text-white shadow-sm transition-all active:scale-[0.98]">
                Applica / Plot
              </Button>
            </div>

            {/* Chart Card */}
            <Card className="border-slate-200 shadow-sm overflow-hidden bg-white">
              <CardHeader className="border-b border-slate-100/50 pb-5 px-6 pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-lg font-semibold tracking-tight text-slate-900">Analisi Segnale</CardTitle>
                    <CardDescription className="text-slate-500 mt-1">acceleration_g vs time_s</CardDescription>
                  </div>
                  <div className="flex items-center gap-5 text-sm bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-100">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-0.5 bg-blue-600/60 rounded-full"></div>
                      <span className="text-slate-600 font-medium">Originale</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-0.5 bg-red-500 rounded-full"></div>
                      <span className="text-slate-600 font-medium">Filtrato</span>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <div className="h-[450px] w-full p-6 pt-8">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={mockData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#f1f5f9" />
                      <XAxis 
                        dataKey="time" 
                        stroke="#94a3b8" 
                        fontSize={12} 
                        tickLine={false} 
                        axisLine={false} 
                        tickMargin={12}
                        minTickGap={30}
                      />
                      <YAxis 
                        stroke="#94a3b8" 
                        fontSize={12} 
                        tickLine={false} 
                        axisLine={false} 
                        tickMargin={12}
                        width={60}
                      />
                      <RechartsTooltip 
                        contentStyle={{ 
                          borderRadius: '8px', 
                          border: '1px solid #e2e8f0', 
                          boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
                          padding: '12px'
                        }}
                        itemStyle={{ fontSize: '13px', fontWeight: 500, padding: '2px 0' }}
                        labelStyle={{ fontSize: '12px', color: '#64748b', marginBottom: '8px', fontWeight: 500 }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="original" 
                        stroke="#3b82f6" 
                        strokeWidth={1.5} 
                        dot={false} 
                        activeDot={{ r: 4, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }}
                        name="Originale"
                        strokeOpacity={0.6}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="filtered" 
                        stroke="#ef4444" 
                        strokeWidth={2} 
                        dot={false} 
                        activeDot={{ r: 4, fill: '#ef4444', stroke: '#fff', strokeWidth: 2 }}
                        name="Filtrato"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

          </div>
        </main>
      </div>
    </div>
  );
}
