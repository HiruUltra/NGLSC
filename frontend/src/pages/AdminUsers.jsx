import { useEffect, useState } from 'react';
export default function AdminUsers() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [editingUser, setEditingUser] = useState(null);
  const [form, setForm] = useState({ name: '', email: '', role: 'Student' });

  const token = localStorage.getItem('token');

  const fetchUsers = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('http://localhost:5000/api/users', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!res.ok) throw new Error('Failed to fetch users');
      const data = await res.json();
      setUsers(data.users || data);
    } catch (err) {
      setError('Failed to fetch users');
      // Fallback: provide mock users so admin UI remains usable offline
      setUsers([
        { id: '1', name: 'Alice Admin', email: 'alice@school.edu', role: 'Admin' },
        { id: '2', name: 'Bob Student', email: 'bob@student.edu', role: 'Student' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchUsers(); }, []);

  const handleDelete = async (id) => {
    if (!confirm('Delete this user?')) return;
    try {
      const res = await fetch(`http://localhost:5000/api/users/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error('Delete failed');
      setUsers(prev => prev.filter(u => u._id !== id && u.id !== id));
    } catch (err) {
      alert('Error deleting user: ' + err.message);
    }
  };

  const startEdit = (user) => {
    setEditingUser(user);
    setForm({ name: user.name || '', email: user.email || '', role: user.role || 'Student' });
  };

  const cancelEdit = () => { setEditingUser(null); setForm({ name: '', email: '', role: 'Student' }); };

  const saveEdit = async () => {
    try {
      const id = editingUser._id || editingUser.id;
      const res = await fetch(`http://localhost:5000/api/users/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error('Update failed');
      const updated = await res.json();
      setUsers(prev => prev.map(u => (u._id === id || u.id === id ? (updated.user || updated) : u)));
      cancelEdit();
    } catch (err) {
      alert('Error updating user: ' + err.message);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-white mb-4">User Management</h1>
          {error && (
            <div className="mb-4 text-red-500 flex items-center gap-4">
              <span>{error}</span>
              <button onClick={fetchUsers} className="px-3 py-1 bg-blue-600 text-white rounded">Retry</button>
            </div>
          )}
          {loading ? (
            <div className="text-gray-200">Loading...</div>
          ) : (
            <div className="bg-transparent rounded-lg shadow overflow-hidden p-2">
              <table className="min-w-full divide-y divide-gray-700 text-gray-200">
                <thead className="bg-gray-800">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Name</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Email</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase">Role</th>
                    <th className="px-6 py-3"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-700">
                  {users.map(user => (
                    <tr key={user._id || user.id} className="bg-gray-800/40">
                      <td className="px-6 py-4 whitespace-nowrap text-gray-100">{user.name}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-gray-200">{user.email}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-gray-200">{user.role}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-right">
                        <button onClick={() => startEdit(user)} className="mr-2 px-3 py-1 bg-yellow-500 text-white rounded">Edit</button>
                        <button onClick={() => handleDelete(user._id || user.id)} className="px-3 py-1 bg-red-600 text-white rounded">Delete</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {editingUser && (
            <div className="mt-6 bg-gray-800 p-4 rounded-lg shadow">
              <h2 className="text-lg font-semibold mb-4 text-white">Edit User</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <input name="name" value={form.name} onChange={e => setForm({...form, name: e.target.value})} className="p-2 border rounded" placeholder="Name" />
                <input name="email" value={form.email} onChange={e => setForm({...form, email: e.target.value})} className="p-2 border rounded" placeholder="Email" />
                <select name="role" value={form.role} onChange={e => setForm({...form, role: e.target.value})} className="p-2 border rounded">
                  <option>Student</option>
                  <option>Admin</option>
                </select>
              </div>
              <div className="mt-4">
                <button onClick={saveEdit} className="px-4 py-2 bg-green-600 text-white rounded mr-2">Save</button>
                <button onClick={cancelEdit} className="px-4 py-2 bg-gray-500 text-white rounded">Cancel</button>
              </div>
            </div>
          )}
    </div>
  );
}
