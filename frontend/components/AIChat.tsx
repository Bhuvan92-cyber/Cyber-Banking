'use client';

import { useState, useEffect } from 'react';
import { Send, Bot, User, Loader2, Sparkles, FileText } from 'lucide-react';
import api from '@/lib/api';
import toast from 'react-hot-toast';

interface Message {
  role: 'user' | 'bot';
  content: string;
}

export default function AIChat() {
  const [datasetName, setDatasetName] = useState<string>('cyber_physical_customers.csv');
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'bot',
      content: 'Welcome to Banking Intelligence RAG. Ask questions about your dataset, e.g.:\n• "What are the top 3 features driving customer churn or defaults?"\n• "Which columns have missing values and what is the distribution?"\n• "Summarize the average balance and transaction count across customers."'
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('last_dataset_name');
      if (saved) {
        setDatasetName(saved);
      }
    }
  }, []);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userText = input.trim();
    const userMsg: Message = { role: 'user', content: userText };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await api.post('/api/chat/', {
        dataset_id: datasetName || 'cyber_physical_customers.csv',
        query: userText,
      });

      if (res.data?.answer) {
        setMessages((prev) => [...prev, { role: 'bot', content: res.data.answer }]);
      } else {
        setMessages((prev) => [...prev, { role: 'bot', content: 'No answer returned from the model.' }]);
      }
    } catch (err: any) {
      const errMsg = err.response?.data?.error || err.response?.data?.hint || 'Query failed. Please verify Ollama is active.';
      toast.error(errMsg);
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          content: `⚠️ Error: ${errMsg}${err.response?.data?.hint ? `\n\nHint: ${err.response.data.hint}` : ''}`
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-6rem)] max-w-5xl mx-auto bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
      {/* Header bar */}
      <div className="px-6 py-4 border-b border-slate-200 bg-slate-50/80 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-600 text-white flex items-center justify-center shadow-sm">
            <Bot size={22} />
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-1.5">
              Dataset RAG Assistant <Sparkles size={14} className="text-amber-500" />
            </h2>
            <p className="text-xs text-slate-500">Grounded schema retrieval powered by LangChain & Ollama</p>
          </div>
        </div>

        {/* Dataset selector input */}
        <div className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-lg border border-slate-200 text-xs">
          <FileText size={14} className="text-slate-400" />
          <span className="text-slate-500 font-medium">Active Dataset:</span>
          <input
            type="text"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            className="font-mono text-blue-700 bg-transparent focus:outline-none w-48 text-xs font-semibold"
            placeholder="dataset.csv"
          />
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-slate-50/30">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`flex items-start gap-3 max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div
                className={`p-2 rounded-xl shrink-0 ${
                  msg.role === 'user' ? 'bg-blue-600 text-white shadow-sm' : 'bg-slate-200 text-slate-700'
                }`}
              >
                {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
              </div>
              <div
                className={`p-4 rounded-2xl text-sm leading-relaxed shadow-sm ${
                  msg.role === 'user'
                    ? 'bg-blue-600 text-white rounded-tr-none'
                    : 'bg-white text-slate-800 border border-slate-200/80 rounded-tl-none'
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="flex items-center gap-3 bg-white px-4 py-3 rounded-2xl border border-slate-200 shadow-sm text-sm text-slate-600">
              <Loader2 className="animate-spin text-blue-600" size={18} />
              <span>Analyzing dataset schema and generating grounded response...</span>
            </div>
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="p-4 border-t border-slate-200 bg-white">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask a question about your uploaded dataset..."
            className="flex-1 px-4 py-3 border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm text-slate-900"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 transition-colors flex items-center justify-center font-medium shadow-sm"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}

