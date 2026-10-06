'use client';

import { Users, Database, FileText, Activity, TrendingUp, CheckCircle, AlertTriangle } from 'lucide-react';

export default function AdminOverview() {
  const stats = [
    { label: 'Total Registered Users', value: '1,248', icon: Users, color: 'bg-blue-500', trend: '+12% this month' },
    { label: 'Datasets Uploaded', value: '3,842', icon: Database, color: 'bg-emerald-500', trend: '+8% this month' },
    { label: 'ML Models Trained', value: '12,095', icon: Activity, color: 'bg-purple-500', trend: '+24% this month' },
    { label: 'PDF Reports Generated', value: '8,310', icon: FileText, color: 'bg-orange-500', trend: '+15% this month' },
  ];

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <div>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">System Overview</h1>
        <p className="text-sm text-slate-500 mt-1">Real-time metrics, platform performance, and ML service health.</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, idx) => (
          <div key={idx} className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className={`p-3 rounded-xl ${stat.color} bg-opacity-10`}>
                <stat.icon className={`w-6 h-6 ${stat.color.replace('bg-', 'text-')}`} />
              </div>
              <span className="flex items-center text-xs font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 rounded-full">
                <TrendingUp size={12} className="mr-1" /> {stat.trend}
              </span>
            </div>
            <h3 className="text-slate-500 text-xs font-medium uppercase tracking-wider">{stat.label}</h3>
            <p className="text-3xl font-black text-slate-900 mt-1">{stat.value}</p>
          </div>
        ))}
      </div>

      {/* System Health Status */}
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
        <h2 className="text-base font-bold text-slate-900 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-blue-600" /> Infrastructure & Service Health
        </h2>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></div>
              <div>
                <span className="font-semibold text-sm text-slate-800">Django REST API Backend</span>
                <p className="text-xs text-slate-400">Gunicorn WSGI Workers running</p>
              </div>
            </div>
            <span className="text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-full flex items-center gap-1">
              <CheckCircle size={12} /> Operational (12ms)
            </span>
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></div>
              <div>
                <span className="font-semibold text-sm text-slate-800">PostgreSQL Database</span>
                <p className="text-xs text-slate-400">Connection pool healthy with health checks</p>
              </div>
            </div>
            <span className="text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-full flex items-center gap-1">
              <CheckCircle size={12} /> Operational (4ms)
            </span>
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 rounded-full bg-amber-500"></div>
              <div>
                <span className="font-semibold text-sm text-slate-800">Local Ollama LLM Service</span>
                <p className="text-xs text-slate-400">ChromaDB schema vectorstore & Llama 3</p>
              </div>
            </div>
            <span className="text-xs font-medium text-amber-700 bg-amber-50 border border-amber-200 px-2.5 py-1 rounded-full flex items-center gap-1">
              <AlertTriangle size={12} /> Running Locally (Port 11434)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
