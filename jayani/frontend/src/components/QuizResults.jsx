/**
 * QuizResults Component
 * Displays quiz results after submission
 */
export default function QuizResults({ result, onRetakeQuiz }) {
    const { score, total, percentage, autoSubmit } = result;

    // Determine grade and message
    const getGradeInfo = () => {
        if (percentage >= 90) {
            return { grade: 'A+', color: 'text-green-400', emoji: '🌟', message: 'Outstanding!' };
        } else if (percentage >= 80) {
            return { grade: 'A', color: 'text-green-300', emoji: '🎉', message: 'Excellent!' };
        } else if (percentage >= 70) {
            return { grade: 'B', color: 'text-blue-400', emoji: '👍', message: 'Good Job!' };
        } else if (percentage >= 60) {
            return { grade: 'C', color: 'text-yellow-400', emoji: '👌', message: 'Well Done!' };
        } else if (percentage >= 50) {
            return { grade: 'D', color: 'text-orange-400', emoji: '📚', message: 'Keep Practicing!' };
        } else {
            return { grade: 'F', color: 'text-red-400', emoji: '💪', message: 'Don\'t Give Up!' };
        }
    };

    const gradeInfo = getGradeInfo();

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 flex items-center justify-center p-6">
            <div className="max-w-2xl w-full">
                {/* Results Card */}
                <div className="bg-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-2xl p-12 shadow-2xl text-center">
                    {/* Header */}
                    {autoSubmit && (
                        <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-xl">
                            <p className="text-yellow-300 text-sm">⏰ Time expired - Quiz auto-submitted</p>
                        </div>
                    )}

                    <div className="text-7xl mb-6">{gradeInfo.emoji}</div>

                    <h1 className="text-4xl font-bold text-white mb-3">
                        Quiz Completed!
                    </h1>

                    <p className="text-gray-300 text-lg mb-8">{gradeInfo.message}</p>

                    {/* Score Display */}
                    <div className="mb-8">
                        <div className={`text-8xl font-bold mb-4 ${gradeInfo.color}`}>
                            {percentage}%
                        </div>
                        <div className="text-2xl text-gray-300">
                            {score} out of {total} correct
                        </div>
                    </div>

                    {/* Grade Badge */}
                    <div className="inline-block mb-8">
                        <div className={`px-8 py-4 rounded-2xl bg-gradient-to-r ${percentage >= 70
                                ? 'from-green-600 to-emerald-600'
                                : percentage >= 50
                                    ? 'from-yellow-600 to-orange-600'
                                    : 'from-red-600 to-pink-600'
                            }`}>
                            <div className="text-sm text-white/80 mb-1">Grade</div>
                            <div className="text-4xl font-bold text-white">{gradeInfo.grade}</div>
                        </div>
                    </div>

                    {/* Statistics */}
                    <div className="grid grid-cols-3 gap-4 mb-8">
                        <div className="p-4 bg-gray-900/50 rounded-xl">
                            <div className="text-3xl font-bold text-green-400">{score}</div>
                            <div className="text-sm text-gray-400 mt-1">Correct</div>
                        </div>
                        <div className="p-4 bg-gray-900/50 rounded-xl">
                            <div className="text-3xl font-bold text-red-400">{total - score}</div>
                            <div className="text-sm text-gray-400 mt-1">Incorrect</div>
                        </div>
                        <div className="p-4 bg-gray-900/50 rounded-xl">
                            <div className="text-3xl font-bold text-blue-400">{total}</div>
                            <div className="text-sm text-gray-400 mt-1">Total</div>
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="space-y-4">
                        <button
                            onClick={onRetakeQuiz}
                            className="w-full py-4 rounded-xl font-bold text-lg bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white transition-all transform hover:scale-105 shadow-lg hover:shadow-purple-500/50"
                        >
                            Take Another Quiz
                        </button>
                    </div>
                </div>

                {/* Footer */}
                <div className="text-center mt-8 text-gray-400 text-sm">
                    <p>Thank you for using AI Exam Proctoring System</p>
                </div>
            </div>
        </div>
    );
}
