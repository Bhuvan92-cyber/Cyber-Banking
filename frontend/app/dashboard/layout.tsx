'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, Upload, MessageSquare, FileText, LogOut, ShieldCheck, Users } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import toast from 'react-hot-toast';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  const handleLogout = async () => {
    await logout();
    toast.success('Logged out successfully');
  };

  const navItems = [
    { href: '/dashboard', label: 'Overview', icon: LayoutDashboard },
    { href: '/dashboard/upload', label: 'Upload Dataset', icon: Upload },
    { href: '/dashboard/chat', label: 'AI Chat (RAG)', icon: MessageSquare },
    { href: '/dashboard/reports', label: 'Predict & Reports', icon: FileText },
  ];

  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 text-white flex flex-col justify-between shrink-0">
        <div>
          <div className="p-6 border-b border-slate-800 flex items-center gap-2">
            <ShieldCheck className="w-6 h-6 text-blue-500" />
            <div>
              <h1 className="text-lg font-bold text-white tracking-tight">CyberBanking <span className="text-xs bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded font-mono">v2</span></h1>
              <p className="text-xs text-slate-400">ML & Analytics Platform</p>
            </div>
          </div>
          
          <nav className="p-4 space-y-1.5">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  <item.icon size={18} />
                  <span>{item.label}</span>
                </Link>
              );
            })}

            {user?.is_staff && (
              <div className="pt-4 mt-4 border-t border-slate-800">
                <Link
                  href="/admin"
                  className="flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium text-amber-400 hover:bg-slate-800 transition-colors"
                >
                  <Users size={18} />
                  <span>Admin Portal</span>
                </Link>
              </div>
            )}
          </nav>
        </div>

        <div className="p-4 border-t border-slate-800">
          <div className="text-xs text-slate-400 mb-1">Logged in as:</div>
          <div className="text-sm font-semibold truncate text-white">{user?.username || 'User'}</div>
          <button
            onClick={handleLogout}
            className="mt-3 flex items-center justify-center gap-2 px-3 py-2 w-full text-xs font-semibold text-slate-300 hover:bg-red-500/20 hover:text-red-400 rounded-lg transition-colors border border-slate-800"
          >
            <LogOut size={16} />
            <span>Sign Out</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-8">
        {children}
      </main>
    </div>
  );
}

