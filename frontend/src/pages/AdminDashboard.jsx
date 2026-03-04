import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';

const AdminDashboard = () => {
    const { token } = useAuth();
    const [analytics, setAnalytics] = useState(null);
    const [violations, setViolations] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 30000); // Refresh every 30s
        return () => clearInterval(interval);
    }, []);

    const fetchData = async () => {
        try {
            const config = { headers: { Authorization: `Bearer ${token}` } };
            const [analyticsRes, violationsRes] = await Promise.all([
                axios.get('http://localhost:8000/api/admin/analytics', config),
                axios.get('http://localhost:8000/api/admin/violations', config)
            ]);

            setAnalytics(analyticsRes.data);
            setViolations(violationsRes.data);
            setError('');
        } catch (err) {
            console.error('Dashboard fetch error:', err);
            setError('Failed to load analytics data.');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-gray-900">
                <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
        );
    }

    return (
        <div className="p-8 bg-gray-900 min-h-screen text-white">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="flex items-center justify-between mb-10">
                    <div>
                        <h1 className="text-4xl font-extrabold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                            Admin Analytics Dashboard
                        </h1>
                        <p className="text-gray-400 mt-2">Real-time student performance & proctoring insights</p>
                    </div>
                    <button
                        onClick={fetchData}
                        className="px-6 py-3 bg-gray-800 hover:bg-gray-700 rounded-xl border border-gray-700 transition-all flex items-center gap-2"
                    >
                        <span>🔄</span> Refresh Data
                    </button>
                </div>

                {error && (
                    <div className="mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-xl text-red-400">
                        {error}
                    </div>
                )}

                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
                    <StatCard
                        title="Average Score"
                        value={`${analytics?.avg_score || 0}%`}
                        icon="📈"
                        color="text-green-400"
                        bg="bg-green-500/10"
                    />
                    <StatCard
                        title="Highest Score"
                        value={`${analytics?.high_score || 0}%`}
                        icon="🏆"
                        color="text-yellow-400"
                        bg="bg-yellow-500/10"
                    />
                    <StatCard
                        title="Total Sessions"
                        value={analytics?.total_quizzes || 0}
                        icon="📝"
                        color="text-blue-400"
                        bg="bg-blue-500/10"
                    />
                    <StatCard
                        title="Integrity Index"
                        value="94%"
                        icon="🛡️"
                        color="text-purple-400"
                        bg="bg-purple-500/10"
                    />
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Violation Table */}
                    <div className="lg:col-span-2 bg-gray-800/50 backdrop-blur-xl rounded-2xl border border-gray-700/50 overflow-hidden">
                        <div className="p-6 border-b border-gray-700/50 flex items-center justify-between">
                            <h2 className="text-xl font-bold flex items-center gap-2">
                                <span>🚨</span> Recent Proctoring Violations
                            </h2>
                            <span className="text-xs bg-red-500/20 text-red-400 px-3 py-1 rounded-full border border-red-500/30 font-medium">
                                Live Monitoring
                            </span>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-left">
                                <thead className="bg-gray-900/50 text-gray-400 text-xs uppercase font-bold">
                                    <tr>
                                        <th className="px-6 py-4">Student</th>
                                        <th className="px-6 py-4">Violation Type</th>
                                        <th className="px-6 py-4">Severity</th>
                                        <th className="px-6 py-4">Time</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-700/30">
                                    {violations.length > 0 ? violations.map((v, i) => (
                                        <tr key={i} className="hover:bg-gray-700/30 transition-colors">
                                            <td className="px-6 py-4 font-medium">{v.username}</td>
                                            <td className="px-6 py-4">
                                                <span className="flex items-center gap-2">
                                                    {v.violation_type === 'TALKING' ? '🗣️' : '👤❓'}
                                                    {v.violation_type.replace('_', ' ')}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className={`px-2 py-1 rounded-md text-[10px] font-bold uppercase ${v.severity === 'critical' ? 'bg-red-500 text-white' : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                                                    }`}>
                                                    {v.severity}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 text-gray-400 text-sm">
                                                {new Date(v.timestamp).toLocaleTimeString()}
                                            </td>
                                        </tr>
                                    )) : (
                                        <tr>
                                            <td colSpan="4" className="px-6 py-12 text-center text-gray-500">
                                                No violations recorded recently. Great work!
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Top Performers */}
                    <div className="bg-gray-800/50 backdrop-blur-xl rounded-2xl border border-gray-700/50 p-6">
                        <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                            <span>✨</span> Top Performers
                        </h2>
                        <div className="space-y-4">
                            {analytics?.top_performers?.map((p, i) => (
                                <div key={i} className="flex items-center justify-between p-4 bg-gray-900/50 rounded-xl border border-gray-700/30">
                                    <div className="flex items-center gap-3">
                                        <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center font-bold text-white shadow-lg">
                                            {p.username[0].toUpperCase()}
                                        </div>
                                        <div>
                                            <p className="font-bold text-gray-100">{p.username}</p>
                                            <p className="text-xs text-gray-400 capitalize">{p.topic}</p>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-cyan-400 font-bold">{p.percentage}%</p>
                                        <p className="text-[10px] text-gray-500">{Math.floor(p.duration_seconds / 60)}m {p.duration_seconds % 60}s</p>
                                    </div>
                                </div>
                            ))}
                            {(!analytics?.top_performers || analytics.top_performers.length === 0) && (
                                <div className="text-center py-8 text-gray-500">
                                    No records yet.
                                </div>
                            )}
                        </div>

                        <div className="mt-8 p-4 bg-cyan-500/10 border border-cyan-500/30 rounded-xl">
                            <h4 className="text-cyan-400 text-sm font-bold flex items-center gap-2 mb-2">
                                <span>🛡️</span> Confidence Peak
                            </h4>
                            <p className="text-xs text-gray-400 leading-relaxed">
                                Students with &lt; 2 violations and &gt; 85% score are
                                categorized as "High Confidence" academic learners.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const StatCard = ({ title, value, icon, color, bg }) => (
    <div className="bg-gray-800/50 backdrop-blur-xl p-6 rounded-2xl border border-gray-700/50 shadow-xl shadow-black/20 group hover:border-cyan-500/50 transition-all">
        <div className="flex items-center justify-between mb-4">
            <span className={`text-2xl p-3 rounded-xl ${bg}`}>{icon}</span>
        </div>
        <p className="text-gray-400 text-sm font-medium">{title}</p>
        <p className={`text-3xl font-extrabold mt-1 ${color}`}>{value}</p>
    </div>
);

export default AdminDashboard;
