import { useState } from 'react';
import WebcamStream from './WebcamStream';

export default function LectureRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [recordings, setRecordings] = useState([
    { id: 1, title: 'Introduction to React', duration: '45 min', date: '2026-01-19', size: '2.4 GB' },
    { id: 2, title: 'Advanced Hooks', duration: '60 min', date: '2026-01-18', size: '3.1 GB' },
    { id: 3, title: 'State Management', duration: '55 min', date: '2026-01-17', size: '2.8 GB' },
  ]);

  const startRecording = () => {
    setIsRecording(true);
  };

  const stopRecording = () => {
    setIsRecording(false);
  };

  return (
    <div className="space-y-6">
      {/* Live Stream Area */}
      <div className="bg-gray-900 rounded-lg overflow-hidden">
        <div className="aspect-video bg-black flex items-center justify-center">
          <WebcamStream />
        </div>
        <div className="p-6 border-t border-gray-700">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h3 className="text-xl font-bold text-white">Live Lecture</h3>
              <p className="text-gray-400 text-sm">Recording Status: {isRecording ? <span className="text-red-500">🔴 Recording</span> : <span className="text-gray-500">⚫ Idle</span>}</p>
            </div>
          </div>
          <div className="flex gap-4">
            <button
              onClick={startRecording}
              disabled={isRecording}
              className="px-6 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
            >
              🔴 Start Recording
            </button>
            <button
              onClick={stopRecording}
              disabled={!isRecording}
              className="px-6 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 disabled:opacity-50"
            >
              ⏹ Stop Recording
            </button>
          </div>
        </div>
      </div>

      {/* Recording List */}
      <div className="bg-gray-900 rounded-lg p-6">
        <h3 className="text-xl font-bold text-white mb-4">Recorded Lectures</h3>
        <div className="space-y-3">
          {recordings.map(recording => (
            <div key={recording.id} className="bg-gray-800 rounded p-4 flex justify-between items-center hover:bg-gray-700 transition">
              <div>
                <h4 className="text-white font-semibold">{recording.title}</h4>
                <p className="text-gray-400 text-sm">{recording.date} • {recording.duration} • {recording.size}</p>
              </div>
              <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                📥 Download
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
