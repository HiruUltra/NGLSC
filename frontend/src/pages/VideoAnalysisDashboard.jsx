export default function VideoAnalysisDashboard() {
  const highlights = [
    { time: '00:15', title: 'Key Concept Introduced', desc: 'Definition of quantum mechanics' },
    { time: '02:30', title: 'Important Formula', desc: 'Schrödinger Equation' },
    { time: '05:45', title: 'Real-world Example', desc: 'Application in electron behavior' },
    { time: '08:20', title: 'Discussion Point', desc: 'Wave-particle duality' }
  ];

  const summary = [
    'Introduced fundamental principles of quantum mechanics',
    'Explained the Schrödinger equation and its applications',
    'Demonstrated wave-particle duality with real-world examples',
    'Discussed the probabilistic nature of quantum particles',
    'Provided practical examples in semiconductor physics'
  ];

  return (
    <div className="p-8">
      <h1 className="text-4xl font-bold text-white mb-2">Intelligent Video Analysis</h1>
      <p className="text-gray-400 mb-8">AI-powered lecture video analysis with auto-generated highlights</p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Video Player */}
        <div className="lg:col-span-2">
          <div className="bg-gray-800 rounded-xl overflow-hidden border border-gray-700">
            <div className="bg-black aspect-video flex items-center justify-center">
              <div className="text-center">
                <div className="text-6xl mb-4">🎬</div>
                <p className="text-gray-400">Lecture Video Player</p>
                <p className="text-gray-500 text-sm mt-2">(Mock - Backend can integrate real player)</p>
              </div>
            </div>

            {/* Timeline */}
            <div className="p-6 border-t border-gray-700">
              <h3 className="text-white font-bold mb-4">📍 Auto-Generated Highlights</h3>
              <div className="space-y-2">
                {highlights.map((highlight, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-4 p-3 bg-gray-700/50 hover:bg-gray-700 rounded-lg cursor-pointer transition"
                  >
                    <span className="text-blue-400 font-bold min-w-12">{highlight.time}</span>
                    <div className="flex-1">
                      <p className="text-white font-medium">{highlight.title}</p>
                      <p className="text-gray-400 text-sm">{highlight.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Summary Panel */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 flex flex-col">
          <h3 className="text-white font-bold mb-4 text-lg">📋 AI Summary</h3>
          <ul className="space-y-3 flex-1">
            {summary.map((point, idx) => (
              <li key={idx} className="flex gap-3">
                <span className="text-blue-400 font-bold flex-shrink-0">✓</span>
                <span className="text-gray-300 text-sm leading-relaxed">{point}</span>
              </li>
            ))}
          </ul>

          {/* Export Button */}
          <button className="w-full mt-6 bg-green-600 hover:bg-green-700 text-white py-3 rounded-lg font-bold transition flex items-center justify-center gap-2">
            📥 Export Report
          </button>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-8">
        {[
          { label: 'Video Duration', value: '45:30' },
          { label: 'Highlights Found', value: '12' },
          { label: 'Key Topics', value: '8' },
          { label: 'Comprehension Score', value: '94%' }
        ].map((metric, idx) => (
          <div key={idx} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <p className="text-gray-400 text-sm">{metric.label}</p>
            <p className="text-3xl font-bold text-white mt-2">{metric.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
