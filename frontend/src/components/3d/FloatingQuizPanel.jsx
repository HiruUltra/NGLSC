import React, { useState, useEffect } from 'react';
import { Html } from '@react-three/drei';

const FloatingQuizPanel = ({ position, quizData, onQuizSubmit, isZoomed, onToggleZoom }) => {
    if (!quizData || !quizData.questions) {
        return (
            <Html center>
                <div className="bg-gray-900/90 p-8 rounded-3xl border border-red-500/50 text-white text-center">
                    <h2 className="text-xl font-bold mb-2">Quiz Data Error</h2>
                    <p className="text-gray-400">Unable to load questions. Please refresh.</p>
                </div>
            </Html>
        );
    }

    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [answers, setAnswers] = useState({});
    const [timeLeft, setTimeLeft] = useState((quizData.duration_minutes || 10) * 60);
    const [isSubmitting, setIsSubmitting] = useState(false);

    useEffect(() => {
        if (timeLeft <= 0) {
            handleSubmit(true);
            return;
        }

        const timer = setInterval(() => {
            setTimeLeft((prev) => prev - 1);
        }, 1000);

        return () => clearInterval(timer);
    }, [timeLeft]);

    const handleAnswerSelect = (questionId, optionIndex) => {
        setAnswers(prev => ({
            ...prev,
            [questionId]: optionIndex
        }));
    };

    const handleNext = () => {
        if (currentQuestion < (quizData.questions?.length || 0) - 1) {
            setCurrentQuestion(prev => prev + 1);
        }
    };

    const handlePrevious = () => {
        if (currentQuestion > 0) {
            setCurrentQuestion(prev => prev - 1);
        }
    };

    const handleSubmit = async (autoSubmit = false) => {
        if (isSubmitting) return;
        setIsSubmitting(true);

        let correct = 0;
        const questions = quizData.questions || [];
        questions.forEach((q) => {
            if (answers[q.id] === q.correct_answer) {
                correct++;
            }
        });

        const total = questions.length || 1;
        const result = {
            answers,
            score: correct,
            total: total,
            percentage: ((correct / total) * 100).toFixed(1),
            timeRemaining: timeLeft,
            autoSubmit
        };

        onQuizSubmit(result);
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const question = quizData.questions[currentQuestion] || { question: '...', options: [] };
    const progress = ((currentQuestion + 1) / (quizData.questions?.length || 1)) * 100;

    return (
        <group position={position}>
            {/* Panel Background Plane for Shadows/Backing */}
            <mesh position={[0, 0, -0.01]}>
                <planeGeometry args={[0.9, 0.6]} />
                <meshStandardMaterial color="#000000" opacity={0.6} transparent />
            </mesh>

            <Html
                transform
                distanceFactor={1.5}
                position={[0, 0, 0]}
                className="w-[600px] select-none pointer-events-auto"
            >
                <div className="bg-gray-900/90 backdrop-blur-xl border border-cyan-500/30 rounded-3xl p-8 shadow-2xl text-white">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-3">
                            <span className="text-2xl">📝</span>
                            <div>
                                <h3 className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                                    {quizData.topic} Quiz
                                </h3>
                                <p className="text-[10px] text-gray-400 uppercase tracking-widest font-bold">Question {currentQuestion + 1} of {quizData.questions.length}</p>
                            </div>
                        </div>

                        <div className="flex items-center gap-4">
                            {/* Zoom/Focus Toggle Button */}
                            <button
                                onClick={onToggleZoom}
                                className={`flex items-center gap-2 px-4 py-2 rounded-xl border transition-all duration-300 font-bold text-sm ${isZoomed
                                    ? 'bg-cyan-500 border-cyan-400 text-white shadow-lg shadow-cyan-500/40'
                                    : 'bg-gray-800 border-gray-700 text-gray-400 hover:text-white hover:border-gray-500'
                                    }`}
                                title={isZoomed ? "Back to normal view" : "Go to Focus mode (Bigger on left)"}
                            >
                                {isZoomed ? '🔍 Normal View' : '🔎 Focus Mode'}
                            </button>

                            <div className={`px-4 py-2 rounded-xl border flex items-center gap-2 font-mono text-xl ${timeLeft < 60 ? 'bg-red-500/20 border-red-500/50 text-red-400 animate-pulse' : 'bg-gray-800 border-gray-700 text-cyan-400'
                                }`}>
                                <span>⏳</span> {formatTime(timeLeft)}
                            </div>
                        </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full h-1.5 bg-gray-800 rounded-full mb-8 overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-cyan-500 to-blue-600 transition-all duration-500 ease-out"
                            style={{ width: `${progress}%` }}
                        />
                    </div>

                    {/* Question */}
                    <div className="min-h-[200px] mb-8">
                        <h2 className="text-2xl font-bold leading-relaxed mb-8">
                            {question.question}
                        </h2>

                        <div className="grid grid-cols-1 gap-4">
                            {question.options.map((option, index) => (
                                <button
                                    key={index}
                                    onClick={() => handleAnswerSelect(question.id, index)}
                                    className={`p-5 rounded-2xl text-left transition-all duration-300 border-2 group flex items-center justify-between ${answers[question.id] === index
                                        ? 'bg-cyan-500/20 border-cyan-500 shadow-lg shadow-cyan-500/20'
                                        : 'bg-gray-800/50 border-gray-700 hover:border-gray-500 hover:bg-gray-800'
                                        }`}
                                >
                                    <span className="flex items-center gap-4">
                                        <span className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold transition-colors ${answers[question.id] === index
                                            ? 'bg-cyan-500 text-white'
                                            : 'bg-gray-700 text-gray-400 group-hover:bg-gray-600 group-hover:text-white'
                                            }`}>
                                            {String.fromCharCode(65 + index)}
                                        </span>
                                        <span className="text-lg font-medium">{option}</span>
                                    </span>
                                    {answers[question.id] === index && (
                                        <div className="w-6 h-6 bg-cyan-500 rounded-full flex items-center justify-center text-white scale-110">
                                            ✓
                                        </div>
                                    )}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Navigation */}
                    <div className="flex items-center justify-between pt-6 border-t border-gray-800">
                        <button
                            onClick={handlePrevious}
                            disabled={currentQuestion === 0}
                            className={`px-8 py-3 rounded-xl font-bold transition-all flex items-center gap-2 ${currentQuestion === 0
                                ? 'opacity-30 cursor-not-allowed text-gray-500'
                                : 'text-gray-300 hover:text-white hover:bg-gray-800'
                                }`}
                        >
                            ← Previous
                        </button>

                        {currentQuestion === quizData.questions.length - 1 ? (
                            <button
                                onClick={() => handleSubmit(false)}
                                disabled={isSubmitting}
                                className="px-10 py-4 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white rounded-xl font-bold shadow-lg shadow-green-500/20 transition-all transform hover:scale-105 active:scale-95 disabled:opacity-50"
                            >
                                {isSubmitting ? 'Submitting...' : 'Finish Exam 🚀'}
                            </button>
                        ) : (
                            <button
                                onClick={handleNext}
                                className="px-10 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white rounded-xl font-bold shadow-lg shadow-cyan-500/20 transition-all transform hover:scale-105 active:scale-95 flex items-center gap-2"
                            >
                                Next →
                            </button>
                        )}
                    </div>
                </div>
            </Html>
        </group>
    );
};

export default FloatingQuizPanel;
