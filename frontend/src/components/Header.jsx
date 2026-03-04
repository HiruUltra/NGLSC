import { Link, useLocation } from 'react-router-dom';
import ThemeToggle from './ThemeToggle';
import { useAuth } from '../context/AuthContext';

function Header() {
    const location = useLocation();
    const { user, logout, token } = useAuth();

    const navLinks = [
        { path: '/home', label: 'Home', icon: '🏠', roles: ['admin', 'student'] },
        { path: '/quiz', label: 'Quiz System', icon: '📝', roles: ['student'] },
        { path: '/lecture-recorder', label: 'Lecture Recorder', icon: '🎥', roles: ['admin'] },
        { path: '/attendance-counter', label: 'Smart Attendance', icon: '🎯', roles: ['admin'] }
    ];

    // Filter links based on user role
    const filteredLinks = navLinks.filter(link =>
        !token || (user && link.roles.includes(user.role))
    );

    return (
        <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 backdrop-blur-xl sticky top-0 z-50 shadow-sm transition-colors duration-300">
            <div className="px-4 sm:px-6 py-3">
                <div className="flex items-center justify-between">
                    {/* Logo and Title */}
                    <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity duration-300">
                        <div className="text-2xl sm:text-3xl">🎓</div>
                        <div>
                            <h1 className="text-base sm:text-lg font-bold text-gray-900 dark:text-white transition-colors duration-300">
                                Next Gen Learning Smart Classroom - NGLSC
                            </h1>
                            <p className="text-gray-600 dark:text-gray-400 text-xs mt-0.5 transition-colors duration-300 hidden sm:block">
                                Advanced Academic Intelligence & Monitoring
                            </p>
                        </div>
                    </Link>

                    {/* Navigation Links */}
                    <nav className="flex gap-2 items-center">
                        {token && filteredLinks.map((link) => (
                            <Link
                                key={link.path}
                                to={link.path}
                                className={`px-3 sm:px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 flex items-center gap-1.5 sm:gap-2 hover:scale-105 ${location.pathname === link.path
                                    ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-md shadow-cyan-500/30'
                                    : 'bg-gray-100 dark:bg-gray-700/50 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600/50 hover:text-gray-900 dark:hover:text-white'
                                    }`}
                            >
                                <span className="text-base">{link.icon}</span>
                                <span className="hidden md:inline text-sm">{link.label}</span>
                            </Link>
                        ))}

                        <div className="h-6 w-[1px] bg-gray-200 dark:bg-gray-700 mx-1 hidden sm:block"></div>

                        {token ? (
                            <div className="flex items-center gap-3 ml-1">
                                <div className="hidden lg:block text-right">
                                    <p className="text-xs font-bold text-gray-900 dark:text-white leading-tight">
                                        {user?.full_name || user?.username}
                                    </p>
                                    <p className="text-[10px] text-gray-500 dark:text-gray-400 uppercase tracking-tighter font-semibold">
                                        {user?.role}
                                    </p>
                                </div>
                                <button
                                    onClick={logout}
                                    className="px-4 py-2 bg-red-50 hover:bg-red-100 dark:bg-red-900/20 dark:hover:bg-red-900/40 text-red-600 dark:text-red-400 text-sm font-bold rounded-xl transition-all duration-300"
                                >
                                    Logout
                                </button>
                            </div>
                        ) : (
                            <div className="flex items-center gap-2">
                                <Link
                                    to="/login"
                                    className="px-4 py-2 text-gray-700 dark:text-gray-300 text-sm font-bold hover:bg-gray-100 dark:hover:bg-gray-700 rounded-xl transition-all duration-300"
                                >
                                    Login
                                </Link>
                                <Link
                                    to="/signup"
                                    className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white text-sm font-bold rounded-xl shadow-lg shadow-cyan-500/30 transition-all duration-300 active:scale-95"
                                >
                                    Sign Up
                                </Link>
                            </div>
                        )}

                        <ThemeToggle />
                    </nav>
                </div>
            </div>
        </header>
    );
}

export default Header;
