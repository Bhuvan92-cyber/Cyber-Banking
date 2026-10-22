'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { LayoutDashboard, Users, Shield, LogOut, ArrowLeft } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import toast from 'react-hot-toast';

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, isAuthenticated, isLoading, logout } = useAuth();

  useEffect(() => {
    if (!isLoading) {
      if (!isAuthenticated) {
        router.replace('/login?redirect=admin');
      } else if (!user?.is_staff) {
        toast.error('Access denied. Staff administrator privileges required.');
        router.replace('/dashboard');
      }
    }
  }, [isAuthenticated, isLoading, user, router]);

  const handleLogout = async () => {
    try {
      await logout();
      toast.success('Logged out successfully');
      router.push('/login?redirect=admin');
    } catch {
      router.push('/login?redirect=admin');
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-900 text-white">
        <div className="flex items-center gap-3">
          <Shield className="w-6 h-6 text-red-500 animate-pulse" />
          <span className="text-sm font-medium">Verifying admin authorization...</span>
        </div>
      </div>
    );
  }

  if (!isAuthenticated || !user?.is_staff) {
    return null;
  }

  const navItems = [
    { href: '/admin', label: 'System Overview', icon: LayoutDashboard },
    { href: '/admin/users', label: 'User Management', icon: Users },
    { href: '/dashboard', label: 'Back to User Dashboard', icon: ArrowLeft, isSecondary: true },
  ];

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Admin Sidebar */}
      <aside className="w-64 bg-slate-900 text-white flex flex-col">
        <div className="p-6 border-b border-slate-700">
          <h1 className="text-xl font-bold text-red-400 flex items-center gap-2">
            <Shield size={20} /> Admin Portal
          </h1>
          <p className="text-xs text-slate-400 mt-1">Welcome, {user?.username || 'Admin'}</p>
        </div>
        
        <nav className="flex-1 p-4 space-y-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                  item.isSecondary 
                    ? 'text-slate-400 hover:bg-slate-800 hover:text-white mt-8 border-t border-slate-700 pt-4'
                    : isActive 
                      ? 'bg-red-600 text-white' 
                      : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                <item.icon size={20} />
                <span className="font-medium">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-slate-700">
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 px-4 py-3 w-full text-slate-300 hover:bg-red-600/20 hover:text-red-400 rounded-lg transition-colors"
          >
            <LogOut size={20} />
            <span className="font-medium">Logout</span>
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

