'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, CheckCircle, FileSpreadsheet, Loader2, Database } from 'lucide-react';
import api from '@/lib/api';
import toast from 'react-hot-toast';

interface DatasetSummary {
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  missing_values: Record<string, number>;
  dtypes: Record<string, string>;
  preview: Array<Record<string, any>>;
}

export default function UploadDataset() {
  const [uploading, setUploading] = useState(false);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [lastUploadedName, setLastUploadedName] = useState<string>('');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await api.post('/api/upload/', formData);
      setSummary(res.data.summary);
      setLastUploadedName(res.data.dataset_id || file.name);
      
      // Save in localStorage so AI Chat and Predict tabs can pick it up automatically
      if (typeof window !== 'undefined') {
        localStorage.setItem('last_dataset_name', res.data.dataset_id || file.name);
      }
      
      toast.success(`Dataset '${file.name}' uploaded and profiled successfully!`);
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false,
  });

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div>
        <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Upload Banking Dataset</h2>
        <p className="text-sm text-slate-500 mt-1">
          Upload customer records or transaction data in CSV format for automated profiling and ML model training.
        </p>
      </div>
      
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
          isDragActive 
            ? 'border-blue-500 bg-blue-50/70 scale-[0.99]' 
            : 'border-slate-300 bg-white hover:border-blue-400 hover:bg-slate-50/50 shadow-sm'
        }`}
      >
        <input {...getInputProps()} />
        <div className="w-16 h-16 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center mx-auto mb-4">
          <UploadCloud className="h-8 w-8" />
        </div>
        <p className="text-base font-semibold text-slate-800">
          {isDragActive ? 'Drop your CSV dataset here...' : 'Drag & drop your CSV file here, or click to browse'}
        </p>
        <p className="text-xs text-slate-500 mt-1.5">Standard .csv files supported (up to 50MB)</p>
      </div>

      {uploading && (
        <div className="flex items-center justify-center gap-3 p-4 bg-blue-50 text-blue-700 rounded-xl border border-blue-200">
          <Loader2 className="animate-spin h-5 w-5" />
          <span className="text-sm font-medium">Processing and profiling dataset with Pandas...</span>
        </div>
      )}

      {summary && (
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-100 pb-4">
            <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
              <CheckCircle className="text-emerald-500 h-5 w-5" /> Dataset Profile: {lastUploadedName}
            </h3>
            <span className="text-xs bg-emerald-50 text-emerald-700 border border-emerald-200 px-2.5 py-1 rounded-full font-medium">
              Profile Ready
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
              <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">Total Rows</p>
              <p className="text-2xl font-extrabold text-slate-900 mt-1">{summary.rows.toLocaleString()}</p>
            </div>
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
              <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">Total Columns</p>
              <p className="text-2xl font-extrabold text-slate-900 mt-1">{summary.columns}</p>
            </div>
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
              <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">Missing Fields</p>
              <p className="text-2xl font-extrabold text-slate-900 mt-1">
                {Object.values(summary.missing_values || {}).reduce((acc, curr) => acc + curr, 0)}
              </p>
            </div>
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
              <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">File Status</p>
              <p className="text-sm font-bold text-emerald-600 mt-2 flex items-center gap-1">
                <Database className="w-4 h-4" /> Validated
              </p>
            </div>
          </div>
          
          <div>
            <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Column Schema</h4>
            <div className="flex flex-wrap gap-2">
              {summary.column_names.map((col: string) => (
                <span key={col} className="px-2.5 py-1 bg-blue-50 text-blue-700 border border-blue-200 text-xs rounded-lg font-mono">
                  {col} <span className="text-blue-400 font-normal">({summary.dtypes?.[col] || 'var'})</span>
                </span>
              ))}
            </div>
          </div>

          {summary.preview && summary.preview.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Dataset Preview (Top Rows)</h4>
              <div className="overflow-x-auto rounded-xl border border-slate-200">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200">
                    <tr>
                      {summary.column_names.map((col) => (
                        <th key={col} className="px-3 py-2.5 whitespace-nowrap">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {summary.preview.map((row, i) => (
                      <tr key={i} className="hover:bg-slate-50">
                        {summary.column_names.map((col) => (
                          <td key={col} className="px-3 py-2 whitespace-nowrap text-slate-600">
                            {row[col] !== null && row[col] !== undefined ? String(row[col]) : '-'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

