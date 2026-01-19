import { Link, useLocation, useNavigate } from 'react-router-dom';
import ThemeToggle from './ThemeToggle';
import { useState } from 'react';

function Header() {
    const location = useLocation();
    const navigate = useNavigate();
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const user = localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')) : null;

    const navLinks = [
        { path: '/home', label: 'Home', icon: '🏠' },
        { path: '/quiz', label: 'Quiz System', icon: '📝' },
        { path: '/assignments', label: 'Smart Assignment', icon: '📊' },
        { path: '/cognivoice', label: 'CogniVoice Viva', icon: '🎤' }
    ];

    const handleLogout = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        navigate('/login');
    };

    const handleProfile = () => {
        navigate('/profile');
        setShowProfileMenu(false);
    };

    return (
        <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 backdrop-blur-xl sticky top-0 z-50 shadow-sm transition-colors duration-300">
            <div className="px-4 sm:px-6 py-3">
                <div className="flex items-center justify-between">
                    {/* Logo and Title */}
                    <Link to="/home" className="flex items-center gap-2 hover:opacity-80 transition-opacity duration-300">
                        <div className="text-2xl sm:text-3xl">🎓</div>
                        <div>
                            <h1 className="text-base sm:text-lg font-bold text-gray-900 dark:text-white transition-colors duration-300">
                                AI Proctoring & Learning System
                            </h1>
                            <p className="text-gray-600 dark:text-gray-400 text-xs mt-0.5 transition-colors duration-300 hidden sm:block">
                                Exam Monitoring & Lecture Recording
                            </p>
                        </div>
                    </Link>

                    {/* Navigation Links */}
                    <nav className="flex gap-2 items-center">
                        {navLinks.map((link) => (
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

                        <ThemeToggle />

                        {/* Profile Menu */}
                        <div className="relative">
                            <button 
                                onClick={() => setShowProfileMenu(!showProfileMenu)}
                                className="w-10 h-10 rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-bold flex items-center justify-center hover:scale-110 transition-all duration-300"
                            >
                                {user?.name ? user.name.charAt(0).toUpperCase() : '👤'}
                            </button>
                            
                            {showProfileMenu && (
                                <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-50">
                                    <div className="p-3 border-b border-gray-200 dark:border-gray-700">
                                        <p className="text-sm font-semibold text-gray-900 dark:text-white">{user?.name || 'User'}</p>
                                        <p className="text-xs text-gray-600 dark:text-gray-400">{user?.email || 'student@example.com'}</p>
                                    </div>
                                    <button
                                        onClick={handleProfile}
                                        className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                                    >
                                        👤 My Profile
                                    </button>
                                    <button
                                        onClick={handleLogout}
                                        className="w-full text-left px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors border-t border-gray-200 dark:border-gray-700"
                                    >
                                        🚪 Logout
                                    </button>
                                </div>
                            )}
                        </div>
                    </nav>
                </div>
            </div>
        </header>
    );
}

export default Header;
