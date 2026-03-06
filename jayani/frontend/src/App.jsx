import { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import ThemeToggle from './components/ThemeToggle';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import QuizConfigScreen from './components/QuizConfigScreen';
import QuizComponent from './components/QuizComponent';
import QuizResults from './components/QuizResults';
import ProctoringWidget from './components/ProctoringWidget';
import LectureRecorderPage from './pages/LectureRecorderPage';
import AttendanceCounterPage from './pages/AttendanceCounterPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ProtectedRoute from './components/ProtectedRoute';
import VoiceQuizPage from "./pages/VoiceQuizPage";
import VoiceQuizResults from "./pages/VoiceQuizResults";
import Profile from "./pages/Profile";

import './App.css';

// Main quiz app component
function QuizApp() {
  const [appState, setAppState] = useState('config');
  const [quizData, setQuizData] = useState(null);
  const [quizResult, setQuizResult] = useState(null);
  const [language, setLanguage] = useState('en');

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
              AI Exam Proctoring System
            </h1>
            <p className="text-gray-600 dark:text-gray-400 text-sm mt-1 transition-colors duration-300">
              Monitored Exam in Progress
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-gray-100 dark:bg-gray-700/50 p-2 rounded-xl border border-gray-200 dark:border-gray-600/50 transition-colors duration-300">
              <button
                onClick={() => setLanguage('en')}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                  language === 'en'
                    ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/30'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                English
              </button>
              <button
                onClick={() => setLanguage('si')}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                  language === 'si'
                    ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/30'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                සිංහල
              </button>
            </div>

            <ThemeToggle />
          </div>
        </div>
      </header>

      <div className="h-[calc(100vh-80px)] flex">
        <div className="flex-1 overflow-hidden bg-white dark:bg-gray-900 transition-colors duration-300">
          <QuizComponent quizData={quizData} onSubmit={handleSubmitQuiz} />
        </div>

        <div className="w-96 p-4 bg-gray-50 dark:bg-gray-800/50 border-l border-gray-200 dark:border-gray-700/50 overflow-y-auto transition-colors duration-300">
          <ProctoringWidget isActive={isExamActive} language={language} />
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          {/* ✅ Auth Routes */}
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          {/* ✅ Protected Routes */}
          <Route
            path="/home"
            element={
              <ProtectedRoute>
                <Layout><HomePage /></Layout>
              </ProtectedRoute>
            }
          />

          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout><HomePage /></Layout>
              </ProtectedRoute>
            }
          />

          <Route
            path="/quiz"
            element={
              <ProtectedRoute>
                <QuizApp />
              </ProtectedRoute>
            }
          />

          <Route
            path="/lecture-recorder"
            element={
              <ProtectedRoute>
                <Layout><LectureRecorderPage /></Layout>
              </ProtectedRoute>
            }
          />

          <Route
            path="/attendance-counter"
            element={
              <ProtectedRoute>
                <Layout><AttendanceCounterPage /></Layout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/voice-quiz"
            element={
              <ProtectedRoute>
                <VoiceQuizPage />
              </ProtectedRoute>
            }
          />

          <Route
            path="/voice-quiz-results"
            element={
              <ProtectedRoute>
                <VoiceQuizResults />
              </ProtectedRoute>
            }
          />
          <Route
            path="/profile"
            element={
              <ProtectedRoute>
                <Profile />
              </ProtectedRoute>
            }
          />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
