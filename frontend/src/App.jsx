import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';

// Auth Pages
import Login from './pages/Login';
import Register from './pages/Register';

// Admin Pages
import AdminDashboard from './pages/AdminDashboard';
import AdminHome from './pages/AdminHome';
import SmartVirtualEnvironment from './pages/SmartVirtualEnvironment';
import VideoAnalysisDashboard from './pages/VideoAnalysisDashboard';
import AdminUsers from './pages/AdminUsers';

// User Pages
import UserHome from './pages/UserHome';
import SmartAssignment from './pages/SmartAssignment';
import CognivoiceViva from './pages/CognivoiceViva';
import QuizConfigScreen from './components/QuizConfigScreen';
import QuizComponent from './components/QuizComponent';
import QuizResults from './components/QuizResults';

// Components
import Layout from './components/Layout';
import ProctoringWidget from './components/ProctoringWidget';
import './App.css';

// Protected Route Component
function ProtectedRoute({ children, requiredRole = null }) {
  const token = localStorage.getItem('token');
  const user = localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')) : null;

  if (!token) {
    return <Navigate to="/login" />;
  }

  if (requiredRole && user?.role !== requiredRole) {
    return <Navigate to={user?.role === 'Admin' ? '/admin/home' : '/home'} />;
  }

  return children;
}

// Quiz App Wrapper
function QuizApp() {
  const [appState, setAppState] = useState('config');
  const [quizData, setQuizData] = useState(null);
  const [quizResult, setQuizResult] = useState(null);

  const handleStartQuiz = (data) => {
    setQuizData(data);
    setAppState('quiz');
  };

  const handleSubmitQuiz = (result) => {
    setQuizResult(result);
    setAppState('results');
  };

  const handleRetakeQuiz = () => {
    setQuizData(null);
    setQuizResult(null);
    setAppState('config');
  };

  const isExamActive = appState === 'quiz';

  if (appState === 'config') {
    return (
      <Layout>
        <QuizConfigScreen onStartQuiz={handleStartQuiz} />
      </Layout>
    );
  }

  if (appState === 'results') {
    return <QuizResults result={quizResult} onRetakeQuiz={handleRetakeQuiz} />;
  }

  return (
    <div className="app-container min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 backdrop-blur-xl shadow-sm transition-colors duration-300">
        <div className="px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2 transition-colors duration-300">
              <span className="text-3xl">🎓</span>
              NGLSC - Quiz System
            </h1>
            <p className="text-gray-600 dark:text-gray-400 text-sm mt-1 transition-colors duration-300">Monitored Exam in Progress</p>
          </div>
        </div>
      </header>

      <div className="h-[calc(100vh-80px)] flex">
        <div className="flex-1 overflow-hidden bg-white dark:bg-gray-900 transition-colors duration-300">
          <QuizComponent
            quizData={quizData}
            onSubmit={handleSubmitQuiz}
          />
        </div>

        <div className="w-96 p-4 bg-gray-50 dark:bg-gray-800/50 border-l border-gray-200 dark:border-gray-700/50 overflow-y-auto transition-colors duration-300">
          <ProctoringWidget
            isActive={isExamActive}
            language="en"
          />
        </div>
      </div>
    </div>
  );
}

// Main App Routes
function AppRoutes() {
  return (
    <Routes>
      {/* Public Routes */}
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />

      {/* Admin Routes */}
      <Route
        path="/admin/*"
        element={
          <ProtectedRoute requiredRole="Admin">
            <Routes>
              <Route element={<AdminDashboard />}>
                <Route path="home" element={<AdminHome />} />
                <Route path="smart-virtual" element={<SmartVirtualEnvironment />} />
                <Route path="video-analysis" element={<VideoAnalysisDashboard />} />
                <Route path="marks" element={<div className="p-8"><p className="text-white text-2xl font-bold">📊 Student Marks Analysis - Coming Soon</p></div>} />
                <Route path="users" element={<AdminUsers />} />
              </Route>
            </Routes>
          </ProtectedRoute>
        }
      />

      {/* User Routes */}
      <Route path="/home" element={<UserHome />} />

      <Route path="/quiz" element={
        <ProtectedRoute>
          <QuizApp />
        </ProtectedRoute>
      } />

      <Route path="/assignments" element={<SmartAssignment />} />

      <Route path="/cognivoice" element={<CognivoiceViva />} />

      {/* Root Route */}
      <Route path="/" element={<Navigate to="/home" />} />
      <Route path="*" element={<Navigate to="/home" />} />
    </Routes>
  );
}

// Main App Component
export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </ThemeProvider>
  );
}

