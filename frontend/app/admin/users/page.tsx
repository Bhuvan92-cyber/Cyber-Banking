'use client';

import { useState } from 'react';
import { Search, MoreVertical, Shield, User, UserCheck, UserX } from 'lucide-react';

const MOCK_USERS = [
  { id: 1, name: 'Bhuvanesh C', email: 'admin@cyberbanking.com', role: 'Superadmin', status: 'Active', joined: '2024-01-10' },
  { id: 2, name: 'Alice Johnson', email: 'alice.johnson@fintech.io', role: 'Staff Analyst', status: 'Active', joined: '2024-02-14' },
  { id: 3, name: 'Bob Smith', email: 'bob.smith@bankcorp.com', role: 'Standard User', status: 'Active', joined: '2024-03-22' },
  { id: 4, name: 'Charlie Brown', email: 'charlie@enterprise.org', role: 'Standard User', status: 'Inactive', joined: '2024-04-05' },
  { id: 5, name: 'Diana Prince', email: 'diana.prince@analytics.co', role: 'Staff Analyst', status: 'Active', joined: '2024-05-18' },
];

export default function UserManagement() {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredUsers = MOCK_USERS.filter((user) =>
    user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.role.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">User Management</h1>
          <p className="text-sm text-slate-500 mt-1">Review active users, role permissions, and access privileges.</p>
        </div>
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
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200 text-xs uppercase">
              <tr>
                <th className="p-4">User</th>
                <th className="p-4">Role</th>
                <th className="p-4">Account Status</th>
                <th className="p-4">Registration Date</th>
                <th className="p-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredUsers.map((user) => (
                <tr key={user.id} className="hover:bg-slate-50/70 transition-colors">
                  <td className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center text-blue-700 font-bold text-sm shadow-sm">
                        {user.name.charAt(0)}
                      </div>
                      <div>
                        <p className="font-semibold text-slate-900">{user.name}</p>
                        <p className="text-slate-500 text-xs">{user.email}</p>
                      </div>
                    </div>
                  </td>
                  <td className="p-4">
                    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${
                      user.role.includes('Admin') 
                        ? 'bg-purple-100 text-purple-700 border border-purple-200' 
                        : user.role.includes('Staff')
                        ? 'bg-blue-100 text-blue-700 border border-blue-200'
                        : 'bg-slate-100 text-slate-700 border border-slate-200'
                    }`}>
                      {user.role.includes('Admin') ? <Shield size={12} /> : <User size={12} />}
                      {user.role}
                    </span>
                  </td>
                  <td className="p-4">
                    <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold ${
                      user.status === 'Active' 
                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                        : 'bg-rose-50 text-rose-700 border border-rose-200'
                    }`}>
                      {user.status === 'Active' ? <UserCheck size={12} /> : <UserX size={12} />}
                      {user.status}
                    </span>
                  </td>
                  <td className="p-4 text-slate-600 font-mono text-xs">{user.joined}</td>
                  <td className="p-4 text-right">
                    <button className="p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors">
                      <MoreVertical size={18} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {filteredUsers.length === 0 && (
          <div className="p-8 text-center text-slate-500 text-sm">
            No registered users found matching "{searchTerm}".
          </div>
        )}
      </div>
    </div>
  );
}
