'use client';

import { useState, useEffect } from 'react';
import { Brain, FileDown, Loader2, Award, CheckCircle2, TrendingUp, BarChart } from 'lucide-react';
import api from '@/lib/api';
import toast from 'react-hot-toast';

interface ModelMetric {
  key: string;
  label: string;
  accuracy: number;
  f1_weighted: number;
  auc: number | null;
  classification_report?: any;
}

interface PredictionResponse {
  mode: string;
  dataset: string;
  best_model: string;
  best_accuracy: number;
  all_models: ModelMetric[];
  note: string;
}

export default function ModelActions() {
  const [datasetName, setDatasetName] = useState<string>('cyber_physical_customers.csv');
  const [predicting, setPredicting] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [results, setResults] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('last_dataset_name');
      if (saved) {
        setDatasetName(saved);
      }
    }
  }, []);

  const handlePredict = async () => {
    if (!datasetName) {
      toast.error('Please enter or upload a dataset first');
      return;
    }
    setPredicting(true);
    try {
      const res = await api.post('/api/predict/', { dataset_name: datasetName });
      setResults(res.data);
      toast.success('Machine learning evaluation complete!');
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Model evaluation failed');
    } finally {
      setPredicting(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!datasetName) {
      toast.error('Please specify a dataset name');
      return;
    }
    setDownloading(true);
    toast.loading('Generating AI-summarized PDF report...', { id: 'pdf-toast' });

    try {
      const response = await api.get(`/api/report/?dataset_name=${datasetName}`, {
        responseType: 'blob',
      });

      // Create download blob
      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `banking_report_${datasetName.replace('.csv', '')}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      toast.success('Report downloaded successfully!', { id: 'pdf-toast' });
    } catch (err: any) {
      toast.error('Failed to generate report. Ensure models have run.', { id: 'pdf-toast' });
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      <div>
        <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Machine Learning & Analytics Reports</h2>
        <p className="text-sm text-slate-500 mt-1">
          Train competitive ML algorithms, evaluate classification metrics, and export board-level PDF reports with AI summaries.
        </p>
      </div>

      {/* Dataset Selector */}
      <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-4">
        <div>
          <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">Target Dataset</label>
          <input
            type="text"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            className="mt-1 font-mono text-sm font-semibold text-blue-700 bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500 w-72"
            placeholder="dataset.csv"
          />
        </div>
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <button
            onClick={handlePredict}
            disabled={predicting}
            className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold shadow-sm transition disabled:opacity-50"
          >
            {predicting ? <Loader2 className="animate-spin h-4 w-4" /> : <Brain className="h-4 w-4" />}
            <span>{predicting ? 'Evaluating Models...' : 'Run Algorithms'}</span>
          </button>

          <button
            onClick={handleDownloadReport}
            disabled={downloading}
            className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-5 py-2.5 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-sm font-semibold shadow-sm transition disabled:opacity-50"
          >
            {downloading ? <Loader2 className="animate-spin h-4 w-4" /> : <FileDown className="h-4 w-4" />}
            <span>Export AI PDF</span>
          </button>
        </div>
      </div>

      {/* Results Display */}
      {results && (
        <div className="space-y-6">
          {/* Top Banner for Best Model */}
          <div className="bg-gradient-to-r from-blue-900 via-indigo-900 to-slate-900 text-white p-6 rounded-2xl shadow-md flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2 text-blue-300 text-xs font-semibold uppercase tracking-wider mb-1">
                <Award size={16} className="text-amber-400" /> Best Performing Algorithm
              </div>
              <h3 className="text-2xl font-extrabold">{results.best_model}</h3>
              <p className="text-xs text-slate-300 mt-1">{results.note}</p>
            </div>
            <div className="text-right">
              <span className="text-xs text-blue-300 font-medium">Top Accuracy</span>
              <p className="text-3xl font-black text-emerald-400">{(results.best_accuracy * 100).toFixed(2)}%</p>
            </div>
          </div>

          {/* Table of all models */}
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="p-5 border-b border-slate-100 flex items-center justify-between">
              <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <BarChart size={18} className="text-blue-600" /> Model Performance Comparison
              </h3>
              <span className="text-xs bg-slate-100 text-slate-600 px-3 py-1 rounded-full font-medium">
                Ranked by Accuracy
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="bg-slate-50 text-slate-600 text-xs uppercase font-semibold border-b border-slate-200">
                  <tr>
                    <th className="px-6 py-3.5">Algorithm</th>
                    <th className="px-6 py-3.5">Accuracy</th>
                    <th className="px-6 py-3.5">F1-Score (Weighted)</th>
                    <th className="px-6 py-3.5">AUC-ROC</th>
                    <th className="px-6 py-3.5 text-right">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {results.all_models?.map((model) => {
                    const isBest = model.label === results.best_model;
                    return (
                      <tr key={model.key} className={isBest ? 'bg-blue-50/50' : 'hover:bg-slate-50/50'}>
                        <td className="px-6 py-4 font-semibold text-slate-900 flex items-center gap-2">
                          {isBest && <CheckCircle2 size={16} className="text-emerald-500" />}
                          {model.label}
                        </td>
                        <td className="px-6 py-4 font-mono font-medium text-slate-700">
                          {(model.accuracy * 100).toFixed(2)}%
                        </td>
                        <td className="px-6 py-4 font-mono text-slate-600">
                          {model.f1_weighted?.toFixed(4) || 'N/A'}
                        </td>
                        <td className="px-6 py-4 font-mono text-slate-600">
                          {model.auc !== null ? model.auc.toFixed(4) : 'N/A'}
                        </td>
                        <td className="px-6 py-4 text-right">
                          <span
                            className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold ${
                              isBest ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-100 text-slate-600'
                            }`}
                          >
                            {isBest ? 'Selected' : 'Trained'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

