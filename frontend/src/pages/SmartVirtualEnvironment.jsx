import { useState } from 'react';
import LectureRecorder from '../components/LectureRecorder';
import AttendanceCounter from '../components/AttendanceCounter';

export default function SmartVirtualEnvironment() {
  const [activeTab, setActiveTab] = useState('recorder');

  return (
    <div className="p-8">
      <h1 className="text-4xl font-bold text-white mb-2">Smart Virtual Environment</h1>
      <p className="text-gray-400 mb-8">Manage lectures and track student attendance</p>

      {/* Tabs */}
      <div className="flex gap-4 mb-8 border-b border-gray-700">
        <button
          onClick={() => setActiveTab('recorder')}
          className={`px-6 py-3 font-semibold transition border-b-2 ${
            activeTab === 'recorder'
              ? 'text-blue-400 border-blue-400'
              : 'text-gray-400 border-transparent hover:text-gray-300'
          }`}
        >
          📹 Lecture Recorder
        </button>
        <button
          onClick={() => setActiveTab('attendance')}
          className={`px-6 py-3 font-semibold transition border-b-2 ${
            activeTab === 'attendance'
              ? 'text-blue-400 border-blue-400'
              : 'text-gray-400 border-transparent hover:text-gray-300'
          }`}
        >
          ✅ Smart Attendance
        </button>
      </div>

      {/* Tab Content */}
      <div>
        {activeTab === 'recorder' && <LectureRecorder />}
        {activeTab === 'attendance' && <AttendanceCounter />}
      </div>
    </div>
  );
}
