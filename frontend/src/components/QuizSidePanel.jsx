import React, { useState, useEffect } from 'react';

const QuizSidePanel = ({ quizData, onQuizSubmit }) => {
    if (!quizData || !quizData.questions) {
        return (
            <div className="h-full bg-gray-900 flex items-center justify-center p-8 border-r border-gray-800 w-[450px]">
                <div className="bg-red-500/10 p-8 rounded-3xl border border-red-500/50 text-white text-center">
                    <h2 className="text-xl font-bold mb-2">Quiz Data Error</h2>
                    <p className="text-gray-400">Unable to load questions. Please refresh.</p>
                </div>
            </div>
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
        <div className="w-[450px] shrink-0 h-full flex flex-col bg-gray-900 border-r border-gray-800 text-white overflow-y-auto">
            <div className="p-6 flex flex-col h-full">
                {/* Header */}
                <div className="flex flex-col gap-4 mb-6">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <span className="text-2xl">📝</span>
                            <div>
                                <h3 className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                                    {quizData.topic} Quiz
                                </h3>
                                <p className="text-[10px] text-gray-400 uppercase tracking-widest font-bold">Question {currentQuestion + 1} of {quizData.questions.length}</p>
                            </div>
                        </div>

                        <div className={`px-4 py-2 rounded-xl border flex items-center justify-center gap-2 font-mono text-xl ${timeLeft < 60 ? 'bg-red-500/20 border-red-500/50 text-red-400 animate-pulse' : 'bg-gray-800 border-gray-700 text-cyan-400'
                            }`}>
                            <span>⏳</span> {formatTime(timeLeft)}
                        </div>
                    </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full h-1.5 bg-gray-800 rounded-full mb-8 overflow-hidden shrink-0">
                    <div
                        className="h-full bg-gradient-to-r from-cyan-500 to-blue-600 transition-all duration-500 ease-out"
                        style={{ width: `${progress}%` }}
                    />
                </div>

                {/* Question */}
                <div className="mb-8">
                    <h2 className="text-xl font-bold leading-relaxed mb-6">
                        {question.question}
                    </h2>

                    <div className="grid grid-cols-1 gap-3">
                        {question.options.map((option, index) => (
                            <button
                                key={index}
                                onClick={() => handleAnswerSelect(question.id, index)}
                                className={`p-4 rounded-2xl text-left transition-all duration-300 border-2 group flex items-start gap-4 ${answers[question.id] === index
                                    ? 'bg-cyan-500/20 border-cyan-500 shadow-md shadow-cyan-500/10'
                                    : 'bg-gray-800/50 border-gray-700 hover:border-gray-500 hover:bg-gray-800'
                                    }`}
                            >
                                <span className={`w-8 h-8 shrink-0 rounded-xl flex items-center justify-center font-bold transition-colors ${answers[question.id] === index
                                    ? 'bg-cyan-500 text-white'
                                    : 'bg-gray-700 text-gray-400 group-hover:bg-gray-600 group-hover:text-white'
                                    }`}>
                                    {String.fromCharCode(65 + index)}
                                </span>
                                <span className="text-[15px] font-medium leading-tight pt-1">{option}</span>
                                {answers[question.id] === index && (
                                    <div className="ml-auto w-5 h-5 bg-cyan-500 rounded-full flex items-center justify-center text-white text-xs shrink-0 mt-1.5">
                                        ✓
                                    </div>
                                )}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Navigation */}
                <div className="flex items-center justify-between pt-6 border-t border-gray-800 mt-auto shrink-0">
                    <button
                        onClick={handlePrevious}
                        disabled={currentQuestion === 0}
                        className={`px-5 py-3 rounded-xl font-bold transition-all flex items-center gap-2 ${currentQuestion === 0
                            ? 'opacity-30 cursor-not-allowed text-gray-500'
                            : 'text-gray-300 hover:text-white hover:bg-gray-800'
                            }`}
                    >
                        ← Prev
                    </button>

                    {currentQuestion === quizData.questions.length - 1 ? (
                        <button
                            onClick={() => handleSubmit(false)}
                            disabled={isSubmitting}
                            className="px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white rounded-xl font-bold shadow-lg shadow-green-500/20 transition-all transform hover:scale-105 active:scale-95 disabled:opacity-50"
                        >
                            {isSubmitting ? '...' : 'Finish Exam 🚀'}
                        </button>
                    ) : (
                        <button
                            onClick={handleNext}
                            className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white rounded-xl font-bold shadow-lg shadow-cyan-500/20 transition-all transform hover:scale-105 active:scale-95 flex items-center gap-2"
                        >
                            Next →
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

export default QuizSidePanel;
