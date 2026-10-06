import Link from 'next/link';
import { FileSpreadsheet, Bot, BarChart3, ArrowRight, ShieldCheck, Zap } from 'lucide-react';

export default function DashboardOverview() {
  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      <div>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">CyberPhysical Analytics Hub</h1>
        <p className="text-sm text-slate-500 mt-1">
          Machine learning pipelines, grounded tabular RAG, and automated compliance reporting.
        </p>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Models Integrated</span>
            <span className="p-2 bg-blue-50 text-blue-600 rounded-lg">
              <Zap size={18} />
            </span>
          </div>
          <p className="text-3xl font-black text-slate-900 mt-4">4</p>
          <p className="text-xs text-slate-500 mt-1">Random Forest, GB, SVM, Logistic Reg</p>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">AI Intelligence</span>
            <span className="p-2 bg-indigo-50 text-indigo-600 rounded-lg">
              <Bot size={18} />
            </span>
          </div>
          <p className="text-3xl font-black text-slate-900 mt-4">Ollama RAG</p>
          <p className="text-xs text-slate-500 mt-1">Local Llama 3 Vector Embeddings</p>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Report Format</span>
            <span className="p-2 bg-sky-50 text-sky-600 rounded-lg">
              <BarChart3 size={18} />
            </span>
          </div>
          <p className="text-3xl font-black text-slate-900 mt-4">ReportLab</p>
          <p className="text-xs text-slate-500 mt-1">Platypus Engine with AI Narrative</p>
        </div>
      </div>

      {/* Quick Action Navigation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link
          href="/dashboard/upload"
          className="group p-6 bg-white rounded-2xl border border-slate-200 shadow-sm hover:border-blue-500 hover:shadow-md transition-all flex flex-col justify-between"
        >
          <div>
            <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center mb-4 group-hover:bg-blue-600 group-hover:text-white transition-colors">
              <FileSpreadsheet size={24} />
            </div>
            <h3 className="text-base font-bold text-slate-900 mb-1">1. Upload Dataset</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Upload customer data in CSV format to trigger automatic profiling and missing field analysis.
            </p>
          </div>
          <div className="mt-6 flex items-center text-xs font-semibold text-blue-600 gap-1 group-hover:translate-x-1 transition-transform">
            Go to Uploader <ArrowRight size={14} />
          </div>
        </Link>

        <Link
          href="/dashboard/chat"
          className="group p-6 bg-white rounded-2xl border border-slate-200 shadow-sm hover:border-indigo-500 hover:shadow-md transition-all flex flex-col justify-between"
        >
          <div>
            <div className="w-12 h-12 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center mb-4 group-hover:bg-indigo-600 group-hover:text-white transition-colors">
              <Bot size={24} />
            </div>
            <h3 className="text-base font-bold text-slate-900 mb-1">2. AI Dataset Chat</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Ask questions about the uploaded dataset using local vector RAG grounded on schema statistics.
            </p>
          </div>
          <div className="mt-6 flex items-center text-xs font-semibold text-indigo-600 gap-1 group-hover:translate-x-1 transition-transform">
            Start Conversation <ArrowRight size={14} />
          </div>
        </Link>

        <Link
          href="/dashboard/reports"
          className="group p-6 bg-white rounded-2xl border border-slate-200 shadow-sm hover:border-sky-500 hover:shadow-md transition-all flex flex-col justify-between"
        >
          <div>
            <div className="w-12 h-12 rounded-xl bg-sky-50 text-sky-600 flex items-center justify-center mb-4 group-hover:bg-sky-600 group-hover:text-white transition-colors">
              <BarChart3 size={24} />
            </div>
            <h3 className="text-base font-bold text-slate-900 mb-1">3. Predict & Reports</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Train algorithms to rank performance and download the PDF report with AI Executive Summary.
            </p>
          </div>
          <div className="mt-6 flex items-center text-xs font-semibold text-sky-600 gap-1 group-hover:translate-x-1 transition-transform">
            Evaluate & Export <ArrowRight size={14} />
          </div>
        </Link>
      </div>
    </div>
  );
}

