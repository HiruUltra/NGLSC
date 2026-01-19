import Header from '../components/Header';

export default function CognivoiceViva() {
  return (
    <>
      <Header />
      <div className="min-h-screen bg-gray-900 p-8">
        <h1 className="text-4xl font-bold text-white mb-2">Cognivoice Viva Assistant</h1>
        <p className="text-gray-400 mb-8">Practice oral exams and presentations with AI voice assistance</p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Main Interface */}
        <div className="bg-gray-800 rounded-xl p-8 border border-gray-700">
          <div className="bg-gray-900 rounded-xl p-12 flex flex-col items-center justify-center mb-6 min-h-96">
            <div className="text-6xl mb-4">🎤</div>
            <p className="text-gray-400 text-center">Click the microphone button to start practicing</p>
            <p className="text-gray-500 text-sm mt-2">Record your answer and get AI feedback</p>
          </div>

          <button className="w-full bg-blue-600 hover:bg-blue-700 text-white py-4 rounded-xl font-bold text-lg transition mb-4">
            🎙️ Start Practice Session
          </button>
          <button className="w-full bg-gray-700 hover:bg-gray-600 text-white py-4 rounded-xl font-bold text-lg transition">
            📋 View Practice History
          </button>
        </div>

        {/* Features */}
        <div className="space-y-4">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-6 text-white">
            <h3 className="text-xl font-bold mb-2">📚 Features</h3>
            <ul className="space-y-2 text-sm">
              <li>✓ Real-time speech recognition</li>
              <li>✓ AI-powered feedback</li>
              <li>✓ Topic-based practice</li>
              <li>✓ Fluency analysis</li>
              <li>✓ Confidence scoring</li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl p-6 text-white">
            <h3 className="text-xl font-bold mb-2">🎯 Topics</h3>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <button className="bg-white/20 hover:bg-white/30 p-2 rounded transition">Physics</button>
              <button className="bg-white/20 hover:bg-white/30 p-2 rounded transition">Chemistry</button>
              <button className="bg-white/20 hover:bg-white/30 p-2 rounded transition">Mathematics</button>
              <button className="bg-white/20 hover:bg-white/30 p-2 rounded transition">Biology</button>
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl p-6 text-white">
            <h3 className="text-xl font-bold mb-2">📊 Your Stats</h3>
            <div className="space-y-2 text-sm">
              <p>Sessions Completed: 12</p>
              <p>Average Score: 82%</p>
              <p>Best Performance: Physics</p>
            </div>
          </div>
        </div>
        </div>
      </div>
    </>
  );
}
