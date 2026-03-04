import { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(localStorage.getItem('token'));
    const [loading, setLoading] = useState(true);

    const API_URL = 'http://localhost:8000';

    useEffect(() => {
        if (token) {
            fetchUser();
        } else {
            setLoading(false);
        }
    }, [token]);

    const fetchUser = async () => {
        try {
            const response = await axios.get(`${API_URL}/auth/me`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setUser(response.data);
        } catch (error) {
            console.error('Failed to fetch user:', error);
            logout();
        } finally {
            setLoading(false);
        }
    };

    const login = async (username, password) => {
        const formData = new FormData();
        formData.append('username', username);
        formData.append('password', password);

        try {
            const response = await axios.post(`${API_URL}/auth/login`, formData);
            const { access_token, role } = response.data;
            localStorage.setItem('token', access_token);
            setToken(access_token);
            return { success: true, role };
        } catch (error) {
            console.error('Login failed:', error);
            return { success: false, message: error.response?.data?.detail || 'Login failed' };
        }
    };

    const signup = async (userData) => {
        try {
            await axios.post(`${API_URL}/auth/register`, userData);
            return { success: true };
        } catch (error) {
            console.error('Signup failed:', error);
            return { success: false, message: error.response?.data?.detail || 'Signup failed' };
        }
    };

    const logout = () => {
        localStorage.removeItem('token');
        setToken(null);
        setUser(null);
    };

    const value = {
        user,
        token,
        loading,
        login,
        signup,
        logout,
        isAdmin: user?.role === 'admin',
        isStudent: user?.role === 'student'
    };

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
