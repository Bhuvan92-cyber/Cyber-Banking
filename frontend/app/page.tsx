import Link from 'next/link';
import { ShieldCheck, Cpu, FileSpreadsheet, Bot, ArrowRight, BarChart3 } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* Navigation */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-xl text-blue-900">
            <ShieldCheck className="w-7 h-7 text-blue-600" />
            <span>CyberPhysicalBanking <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-semibold">v2</span></span>
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/login"
              className="text-sm font-semibold text-slate-600 hover:text-blue-600 transition"
            >
              Sign In
            </Link>
            <Link
              href="/register"
              className="text-sm font-semibold bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition shadow-sm"
            >
              Get Started
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-6 pt-20 pb-16 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold mb-6">
          <Cpu className="w-4 h-4" /> Next-Generation Banking AI & Machine Learning
        </div>
        <h1 className="text-4xl sm:text-6xl font-extrabold text-slate-950 tracking-tight max-w-4xl mx-auto leading-tight">
          Enterprise Financial Intelligence Driven by <span className="text-blue-600">Local RAG & AutoML</span>
        </h1>
        <p className="mt-6 text-lg sm:text-xl text-slate-600 max-w-2xl mx-auto">
          Analyze customer churn, detect fraud, forecast defaults, and query your tabular data in plain English with Ollama and LangChain.
        </p>
        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <Link
            href="/register"
            className="flex items-center gap-2 bg-blue-600 text-white text-base font-semibold px-6 py-3.5 rounded-xl hover:bg-blue-700 shadow-lg shadow-blue-500/20 transition"
          >
            Launch Platform <ArrowRight className="w-5 h-5" />
          </Link>
          <Link
            href="/login"
            className="flex items-center gap-2 bg-white text-slate-700 border border-slate-300 text-base font-semibold px-6 py-3.5 rounded-xl hover:bg-slate-100 transition shadow-sm"
          >
            Sign In to Dashboard
          </Link>
        </div>
      </section>

      {/* Feature Grid */}
      <section className="max-w-7xl mx-auto px-6 py-16">
        <div className="grid md:grid-cols-3 gap-8">
          <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition">
            <div className="w-12 h-12 rounded-xl bg-blue-50 flex items-center justify-center text-blue-600 mb-6">
              <FileSpreadsheet className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">Automated Data Profiling</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Upload customer and transaction datasets. Instantly view schema types, missing value distributions, and statistical summaries.
            </p>
          </div>

          <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition">
            <div className="w-12 h-12 rounded-xl bg-indigo-50 flex items-center justify-center text-indigo-600 mb-6">
              <BarChart3 className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">Multi-Model ML Engine</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Trains and evaluates Random Forest, Gradient Boosting, SVM, and Logistic Regression with SMOTE imbalance treatment.
            </p>
          </div>

          <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition">
            <div className="w-12 h-12 rounded-xl bg-sky-50 flex items-center justify-center text-sky-600 mb-6">
              <Bot className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">Dataset RAG & Executive Reports</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Interact with your data through natural language and generate boardroom-ready PDF reports with AI-crafted Executive Summaries.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
