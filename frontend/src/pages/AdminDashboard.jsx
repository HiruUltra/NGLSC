import { useState } from 'react';
import { useNavigate, Outlet, Link, useLocation } from 'react-router-dom';

export default function AdminDashboard() {
  const navigate = useNavigate();
  const location = useLocation();
  const [user, setUser] = useState(() => {
    const stored = localStorage.getItem('user');
    return stored ? JSON.parse(stored) : null;
  });
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/login');
  };

  const menuItems = [
    { id: 'home', label: 'Dashboard', icon: '📊', path: '/admin/home' },
    { id: 'virtual', label: 'Smart Virtual Environment', icon: '🎓', path: '/admin/smart-virtual' },
    { id: 'video', label: 'Intelligent Video Analysis', icon: '📹', path: '/admin/video-analysis' },
    { id: 'marks', label: 'Student Marks Analysis', icon: '📈', path: '/admin/marks' },
    { id: 'users', label: 'User Management', icon: '👥', path: '/admin/users' }
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <div className="flex h-screen bg-gray-900">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-20'} bg-gray-800 border-r border-gray-700 transition-all duration-300 flex flex-col`}>
        {/* Logo */}
        <div className="p-6 border-b border-gray-700">
          <div className="text-2xl font-bold text-white">
            {sidebarOpen ? '🎓 NGLSC' : '📚'}
          </div>
          {sidebarOpen && <p className="text-xs text-gray-400 mt-1">Admin Portal</p>}
        </div>

        {/* Menu Items */}
        <nav className="flex-1 p-4 space-y-2">
          {menuItems.map(item => (
            <Link
              key={item.id}
              to={item.path}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                isActive(item.path)
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:bg-gray-700'
              }`}
            >
              <span className="text-xl">{item.icon}</span>
              {sidebarOpen && <span className="font-medium">{item.label}</span>}
            </Link>
          ))}
        </nav>

        {/* Toggle Button */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="mx-4 mb-4 p-3 bg-gray-700 hover:bg-gray-600 rounded-lg text-white transition"
        >
          {sidebarOpen ? '◀️' : '▶️'}
        </button>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Header */}
        <header className="bg-gray-800 border-b border-gray-700 px-8 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">NGLSC Admin Dashboard</h1>
            <p className="text-gray-400 text-sm">Welcome back, {user?.name}</p>
          </div>

          <div className="flex items-center gap-6">
            <span className="text-gray-300">{user?.email}</span>
            <button
              onClick={handleLogout}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition"
            >
              Logout
            </button>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto bg-gray-900">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
