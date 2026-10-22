'use client';

import { useState, useEffect, useRef } from 'react';
import { 
  Search, 
  MoreVertical, 
  Shield, 
  User, 
  UserCheck, 
  UserX, 
  Trash2, 
  RefreshCw, 
  AlertTriangle, 
  X, 
  Loader2,
  CheckCircle2,
  ShieldAlert
} from 'lucide-react';
import api from '@/lib/api';
import { useAuth } from '@/context/AuthContext';
import toast from 'react-hot-toast';

export interface AdminUser {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  name: string;
  role: 'Superadmin' | 'Staff Analyst' | 'Standard User';
  status: 'Active' | 'Inactive';
  is_staff: boolean;
  is_superuser: boolean;
  is_active: boolean;
  joined: string;
  date_joined: string;
}

export default function UserManagement() {
  const { user: currentUser } = useAuth();
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  
  // Action Dropdown State
  const [activeDropdownId, setActiveDropdownId] = useState<number | null>(null);
  const dropdownRef = useRef<HTMLDivElement | null>(null);

  // Role Modal State
  const [roleModalUser, setRoleModalUser] = useState<AdminUser | null>(null);
  const [selectedRole, setSelectedRole] = useState<'Superadmin' | 'Staff Analyst' | 'Standard User'>('Standard User');
  const [isUpdatingRole, setIsUpdatingRole] = useState(false);

  // Delete Modal State
  const [deleteModalUser, setDeleteModalUser] = useState<AdminUser | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Status Toggle Loading ID
  const [statusTogglingId, setStatusTogglingId] = useState<number | null>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setActiveDropdownId(null);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Fetch users from API
  const fetchUsers = async (showLoading = true) => {
    if (showLoading) setIsLoading(true);
    else setIsRefreshing(true);

    try {
      const res = await api.get('/api/admin/users/');
      setUsers(res.data);
    } catch (err: any) {
      const errorMsg = err.response?.data?.error || err.response?.data?.detail || 'Failed to fetch registered users';
      toast.error(errorMsg);
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchUsers(true);
  }, []);

  // Open Role Modal
  const openRoleModal = (user: AdminUser) => {
    setRoleModalUser(user);
    setSelectedRole(user.role);
    setActiveDropdownId(null);
  };

  // Submit Role Change
  const handleRoleSubmit = async () => {
    if (!roleModalUser) return;
    setIsUpdatingRole(true);
    try {
      const res = await api.patch(`/api/admin/users/${roleModalUser.id}/`, {
        role: selectedRole,
      });
      toast.success(res.data?.message || `Updated ${roleModalUser.username}'s role to ${selectedRole}`);
      setRoleModalUser(null);
      await fetchUsers(false);
    } catch (err: any) {
      const msg = err.response?.data?.error || 'Failed to update user role';
      toast.error(msg);
    } finally {
      setIsUpdatingRole(false);
    }
  };

  // Toggle Account Active / Inactive Status
  const handleToggleStatus = async (user: AdminUser) => {
    setActiveDropdownId(null);
    if (user.id === currentUser?.id) {
      toast.error('You cannot deactivate your own administrative account.');
      return;
    }

    const nextStatus = !user.is_active;
    setStatusTogglingId(user.id);
    try {
      const res = await api.patch(`/api/admin/users/${user.id}/`, {
        is_active: nextStatus,
      });
      toast.success(res.data?.message || `User '${user.username}' is now ${nextStatus ? 'Active' : 'Inactive'}`);
      await fetchUsers(false);
    } catch (err: any) {
      const msg = err.response?.data?.error || 'Failed to toggle account status';
      toast.error(msg);
    } finally {
      setStatusTogglingId(null);
    }
  };

  // Open Delete Modal
  const openDeleteModal = (user: AdminUser) => {
    setActiveDropdownId(null);
    if (user.id === currentUser?.id) {
      toast.error('You cannot delete your own administrative account.');
      return;
    }
    setDeleteModalUser(user);
  };

  // Confirm Delete User
  const handleDeleteConfirm = async () => {
    if (!deleteModalUser) return;
    setIsDeleting(true);
    try {
      const res = await api.delete(`/api/admin/users/${deleteModalUser.id}/`);
      toast.success(res.data?.message || `User '${deleteModalUser.username}' permanently deleted`);
      setDeleteModalUser(null);
      await fetchUsers(false);
    } catch (err: any) {
      const msg = err.response?.data?.error || 'Failed to delete user';
      toast.error(msg);
    } finally {
      setIsDeleting(false);
    }
  };

  // Filter users based on search
  const filteredUsers = users.filter((user) => {
    const term = searchTerm.toLowerCase();
    const nameMatch = (user.name || '').toLowerCase().includes(term);
    const usernameMatch = (user.username || '').toLowerCase().includes(term);
    const emailMatch = (user.email || '').toLowerCase().includes(term);
    const roleMatch = (user.role || '').toLowerCase().includes(term);
    const statusMatch = (user.status || '').toLowerCase().includes(term);
    return nameMatch || usernameMatch || emailMatch || roleMatch || statusMatch;
  });

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header & Search */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2.5">
            User Management & RBAC
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Manage user roles, modify platform permissions, and enforce administrative access control.
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
            <input
              type="text"
              placeholder="Search by name, email, or role..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 bg-white border border-slate-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 w-full sm:w-80 shadow-sm"
            />
          </div>

          <button
            onClick={() => fetchUsers(false)}
            disabled={isRefreshing || isLoading}
            title="Refresh user list"
            className="p-2.5 bg-white border border-slate-300 rounded-xl text-slate-600 hover:text-slate-900 hover:bg-slate-50 transition shadow-sm disabled:opacity-50"
          >
            <RefreshCw size={18} className={isRefreshing ? 'animate-spin text-blue-600' : ''} />
          </button>
        </div>
      </div>

      {/* Main Table */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="overflow-x-auto min-h-[300px]">
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200 text-xs uppercase tracking-wider">
              <tr>
                <th className="p-4">User</th>
                <th className="p-4">Role</th>
                <th className="p-4">Account Status</th>
                <th className="p-4">Registration Date</th>
                <th className="p-4 text-right">Actions</th>
              </tr>
            </thead>
            
            <tbody className="divide-y divide-slate-100">
              {isLoading ? (
                <tr>
                  <td colSpan={5} className="p-12 text-center text-slate-500">
                    <div className="flex flex-col items-center justify-center gap-2">
                      <Loader2 size={24} className="animate-spin text-blue-600" />
                      <span className="text-xs font-medium">Loading user profiles from DRF backend...</span>
                    </div>
                  </td>
                </tr>
              ) : filteredUsers.map((user) => {
                const isSelf = user.id === currentUser?.id;
                const isDropdownOpen = activeDropdownId === user.id;

                return (
                  <tr key={user.id} className="hover:bg-slate-50/70 transition-colors">
                    {/* User Identity */}
                    <td className="p-4">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-sm shadow-sm ${
                          user.role === 'Superadmin'
                            ? 'bg-purple-100 text-purple-700'
                            : user.role === 'Staff Analyst'
                            ? 'bg-blue-100 text-blue-700'
                            : 'bg-slate-100 text-slate-700'
                        }`}>
                          {(user.name || user.username || '?').charAt(0).toUpperCase()}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <p className="font-semibold text-slate-900">{user.name || user.username}</p>
                            {isSelf && (
                              <span className="text-[10px] bg-blue-100 text-blue-700 font-semibold px-2 py-0.5 rounded-full">
                                You
                              </span>
                            )}
                          </div>
                          <p className="text-slate-500 text-xs font-mono">{user.email || user.username}</p>
                        </div>
                      </div>
                    </td>

                    {/* Role Badge */}
                    <td className="p-4">
                      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${
                        user.role === 'Superadmin'
                          ? 'bg-purple-100 text-purple-700 border border-purple-200' 
                          : user.role === 'Staff Analyst'
                          ? 'bg-blue-100 text-blue-700 border border-blue-200'
                          : 'bg-slate-100 text-slate-700 border border-slate-200'
                      }`}>
                        {user.role === 'Superadmin' ? (
                          <ShieldAlert size={13} />
                        ) : user.role === 'Staff Analyst' ? (
                          <Shield size={13} />
                        ) : (
                          <User size={13} />
                        )}
                        {user.role}
                      </span>
                    </td>

                    {/* Account Status Badge */}
                    <td className="p-4">
                      {statusTogglingId === user.id ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-500">
                          <Loader2 size={12} className="animate-spin text-blue-600" /> Updating...
                        </span>
                      ) : (
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${
                          user.status === 'Active' 
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                            : 'bg-rose-50 text-rose-700 border border-rose-200'
                        }`}>
                          {user.status === 'Active' ? <UserCheck size={13} /> : <UserX size={13} />}
                          {user.status}
                        </span>
                      )}
                    </td>

                    {/* Registration Date */}
                    <td className="p-4 text-slate-600 font-mono text-xs">
                      {user.joined || (user.date_joined ? user.date_joined.slice(0, 10) : '—')}
                    </td>

                    {/* Actions Menu */}
                    <td className="p-4 text-right relative">
                      <div className="inline-block text-left" ref={isDropdownOpen ? dropdownRef : null}>
                        <button
                          onClick={() => setActiveDropdownId(isDropdownOpen ? null : user.id)}
                          className="p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
                          title="Open action menu"
                        >
                          <MoreVertical size={18} />
                        </button>

                        {/* Dropdown Popover */}
                        {isDropdownOpen && (
                          <div className="absolute right-0 mt-2 w-56 bg-white rounded-xl shadow-lg border border-slate-200 py-1.5 z-30 focus:outline-none animate-in fade-in zoom-in-95 duration-100">
                            <div className="px-3.5 py-2 border-b border-slate-100">
                              <p className="text-xs font-semibold text-slate-800 truncate">{user.username}</p>
                              <p className="text-[11px] text-slate-400 truncate">{user.role}</p>
                            </div>

                            {/* Change Role Option */}
                            <button
                              onClick={() => openRoleModal(user)}
                              className="w-full text-left px-3.5 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 flex items-center gap-2.5 transition"
                            >
                              <Shield size={14} className="text-blue-600" />
                              <span>Change Role Permissions</span>
                            </button>

                            {/* Toggle Active / Inactive Status */}
                            <button
                              onClick={() => handleToggleStatus(user)}
                              disabled={isSelf}
                              className={`w-full text-left px-3.5 py-2 text-xs font-medium flex items-center gap-2.5 transition ${
                                isSelf
                                  ? 'text-slate-300 cursor-not-allowed'
                                  : user.is_active
                                  ? 'text-amber-700 hover:bg-amber-50'
                                  : 'text-emerald-700 hover:bg-emerald-50'
                              }`}
                            >
                              {user.is_active ? (
                                <>
                                  <UserX size={14} className={isSelf ? 'text-slate-300' : 'text-amber-600'} />
                                  <span>Deactivate Account</span>
                                </>
                              ) : (
                                <>
                                  <UserCheck size={14} className="text-emerald-600" />
                                  <span>Activate Account</span>
                                </>
                              )}
                            </button>

                            <div className="border-t border-slate-100 my-1"></div>

                            {/* Delete User Option */}
                            <button
                              onClick={() => openDeleteModal(user)}
                              disabled={isSelf}
                              className={`w-full text-left px-3.5 py-2 text-xs font-medium flex items-center gap-2.5 transition ${
                                isSelf 
                                  ? 'text-slate-300 cursor-not-allowed' 
                                  : 'text-rose-600 hover:bg-rose-50'
                              }`}
                            >
                              <Trash2 size={14} className={isSelf ? 'text-slate-300' : 'text-rose-600'} />
                              <span>Delete Permanently</span>
                            </button>
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Empty State */}
        {!isLoading && filteredUsers.length === 0 && (
          <div className="p-12 text-center text-slate-500 text-sm">
            No registered users found matching "{searchTerm}".
          </div>
        )}
      </div>

      {/* ────────────────────────────────────────────────────────────────────────── */}
      {/* Role Modification Modal */}
      {/* ────────────────────────────────────────────────────────────────────────── */}
      {roleModalUser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-150">
          <div className="bg-white rounded-2xl max-w-md w-full shadow-2xl border border-slate-200 overflow-hidden">
            <div className="p-6 border-b border-slate-100 flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-slate-900">Modify Role Permissions</h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Assign platform access tier for <strong>{roleModalUser.username}</strong>
                </p>
              </div>
              <button
                onClick={() => setRoleModalUser(null)}
                className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition"
              >
                <X size={18} />
              </button>
            </div>

            <div className="p-6 space-y-3">
              {[
                {
                  role: 'Standard User' as const,
                  title: 'Standard User',
                  desc: 'General access to analytics hub, dataset upload, ML prediction, RAG chat, and report downloads.',
                  icon: User,
                  color: 'border-slate-300 text-slate-700',
                  activeColor: 'border-blue-600 bg-blue-50/50 text-blue-900',
                },
                {
                  role: 'Staff Analyst' as const,
                  title: 'Staff Analyst',
                  desc: 'Elevated staff privileges. Access to system analytics and admin overview dashboard.',
                  icon: Shield,
                  color: 'border-slate-300 text-slate-700',
                  activeColor: 'border-indigo-600 bg-indigo-50/50 text-indigo-900',
                },
                {
                  role: 'Superadmin' as const,
                  title: 'Superadmin',
                  desc: 'Full administrative control, user RBAC management, permissions mutation, and system configurations.',
                  icon: ShieldAlert,
                  color: 'border-slate-300 text-slate-700',
                  activeColor: 'border-purple-600 bg-purple-50/50 text-purple-900',
                },
              ].map((tier) => {
                const isSelected = selectedRole === tier.role;
                const IconComponent = tier.icon;

                return (
                  <div
                    key={tier.role}
                    onClick={() => setSelectedRole(tier.role)}
                    className={`p-4 rounded-xl border-2 cursor-pointer transition flex items-start gap-3.5 ${
                      isSelected ? tier.activeColor : 'border-slate-200 hover:border-slate-300 bg-white'
                    }`}
                  >
                    <div className={`p-2 rounded-lg mt-0.5 ${
                      isSelected ? 'bg-white shadow-sm' : 'bg-slate-100'
                    }`}>
                      <IconComponent size={18} className={isSelected ? 'text-blue-600' : 'text-slate-500'} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-bold">{tier.title}</span>
                        {isSelected && <CheckCircle2 size={16} className="text-blue-600" />}
                      </div>
                      <p className="text-xs text-slate-500 mt-1 leading-relaxed">{tier.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="p-6 bg-slate-50 border-t border-slate-100 flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={() => setRoleModalUser(null)}
                disabled={isUpdatingRole}
                className="px-4 py-2 text-xs font-semibold text-slate-600 hover:text-slate-800 hover:bg-slate-200/60 rounded-xl transition"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleRoleSubmit}
                disabled={isUpdatingRole}
                className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-xl shadow-sm transition disabled:opacity-50"
              >
                {isUpdatingRole ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    <span>Updating Permissions...</span>
                  </>
                ) : (
                  <span>Save Role Permissions</span>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ────────────────────────────────────────────────────────────────────────── */}
      {/* Permanent Delete Confirmation Modal */}
      {/* ────────────────────────────────────────────────────────────────────────── */}
      {deleteModalUser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-150">
          <div className="bg-white rounded-2xl max-w-md w-full shadow-2xl border border-slate-200 overflow-hidden">
            <div className="p-6 border-b border-slate-100 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-rose-50 text-rose-600 border border-rose-200">
                  <AlertTriangle size={20} />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">Delete User Account</h3>
                  <p className="text-xs text-slate-500">Irreversible action</p>
                </div>
              </div>
              <button
                onClick={() => setDeleteModalUser(null)}
                className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition"
              >
                <X size={18} />
              </button>
            </div>

            <div className="p-6 space-y-3">
              <p className="text-sm text-slate-600 leading-relaxed">
                Are you sure you want to permanently delete user account{' '}
                <strong className="text-slate-900 font-semibold">{deleteModalUser.username}</strong> ({deleteModalUser.email || 'No email'})?
              </p>
              <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800 leading-relaxed">
                ⚠️ All associated authentication credentials and active sessions will be instantly purged. This action cannot be undone.
              </div>
            </div>

            <div className="p-6 bg-slate-50 border-t border-slate-100 flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={() => setDeleteModalUser(null)}
                disabled={isDeleting}
                className="px-4 py-2 text-xs font-semibold text-slate-600 hover:text-slate-800 hover:bg-slate-200/60 rounded-xl transition"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDeleteConfirm}
                disabled={isDeleting}
                className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-rose-600 hover:bg-rose-700 rounded-xl shadow-sm transition disabled:opacity-50"
              >
                {isDeleting ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    <span>Deleting User...</span>
                  </>
                ) : (
                  <>
                    <Trash2 size={14} />
                    <span>Delete Permanently</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
