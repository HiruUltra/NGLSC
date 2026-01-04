import { useState } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
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
import './App.css';

// Main quiz app component
function QuizApp() {
    const [appState, setAppState] = useState('config'); // 'config', 'quiz', 'results'
    const [quizData, setQuizData] = useState(null);
    const [quizResult, setQuizResult] = useState(null);
    const [language, setLanguage] = useState('en'); // 'en' or 'si'

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

    // Only show proctoring during quiz
    const isExamActive = appState === 'quiz';

    // Config screen (no proctoring)
    if (appState === 'config') {
        return (
            <Layout>
                <QuizConfigScreen onStartQuiz={handleStartQuiz} />
            </Layout>
        );
    }

    // Results screen (no proctoring)
    if (appState === 'results') {
        return <QuizResults result={quizResult} onRetakeQuiz={handleRetakeQuiz} />;
    }

    // Quiz screen (with proctoring)
    return (
        <div className="app-container min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
            {/* Header */}
            <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 backdrop-blur-xl shadow-sm transition-colors duration-300">
                <div className="px-6 py-4 flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2 transition-colors duration-300">
                            <span className="text-3xl">🎓</span>
                            AI Exam Proctoring System
                        </h1>
                        <p className="text-gray-600 dark:text-gray-400 text-sm mt-1 transition-colors duration-300">Monitored Exam in Progress</p>
                    </div>

                    <div className="flex items-center gap-3">
                        {/* Language Toggle */}
                        <div className="flex items-center gap-2 bg-gray-100 dark:bg-gray-700/50 p-2 rounded-xl border border-gray-200 dark:border-gray-600/50 transition-colors duration-300">
                            <button
                                onClick={() => setLanguage('en')}
                                className={`px-3 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${language === 'en'
                                    ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/30'
                                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                                    }`}
                            >
                                English
                            </button>
                            <button
                                onClick={() => setLanguage('si')}
                                className={`px-3 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${language === 'si'
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

            {/* Main Content Area */}
            <div className="h-[calc(100vh-80px)] flex">
                {/* Quiz Area - Takes most of the space */}
                <div className="flex-1 overflow-hidden bg-white dark:bg-gray-900 transition-colors duration-300">
                    <QuizComponent
                        quizData={quizData}
                        onSubmit={handleSubmitQuiz}
                    />
                </div>

                {/* Proctoring Sidebar */}
                <div className="w-96 p-4 bg-gray-50 dark:bg-gray-800/50 border-l border-gray-200 dark:border-gray-700/50 overflow-y-auto transition-colors duration-300">
                    <ProctoringWidget
                        isActive={isExamActive}
                        language={language}
                    />
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
                    {/* Home Page Route */}
                    <Route path="/home" element={<Layout><HomePage /></Layout>} />

                    {/* Default route redirects to home */}
                    <Route path="/" element={<Layout><HomePage /></Layout>} />

                    {/* Quiz System Route */}
                    <Route path="/quiz" element={<QuizApp />} />

                    {/* Lecture Recorder Route */}
                    <Route
                        path="/lecture-recorder"
                        element={<Layout><LectureRecorderPage /></Layout>}
                    />

                    {/* Attendance Counter Route */}
                    <Route
                        path="/attendance-counter"
                        element={<Layout><AttendanceCounterPage /></Layout>}
                    />
                </Routes>
            </BrowserRouter>
        </ThemeProvider>
    );
}

export default App;

