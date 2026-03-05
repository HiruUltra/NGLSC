import { useState, useEffect } from 'react';
import WebcamStream from './WebcamStream';
import AlertDisplay from './AlertDisplay';
import StatusMonitor from './StatusMonitor';
import AudioAlert from './AudioAlert';
import { useAuth } from '../context/AuthContext';
import { useWebSocket } from '../hooks/useWebSocket';

/**
 * ProctoringWidget Component
 * Sidebar widget that handles all proctoring features
 * Only active when isActive prop is true
 */
export default function ProctoringWidget({
    isActive,
    language = 'en',
    externalAlert,
    isConnected,
    sendMessage,
    lastMessage,
    error
}) {
    const [currentAlert, setCurrentAlert] = useState(null);
    const [status, setStatus] = useState(null);
    const [violationHistory, setViolationHistory] = useState([]);

    // Sync external alert to local display
    useEffect(() => {
        if (externalAlert) {
            setCurrentAlert(externalAlert);
            if (externalAlert.severity === 'critical' || externalAlert.severity === 'warning') {
                setViolationHistory(prev => [...prev, externalAlert]);
            }
            // Alert auto-clears in App.jsx state, but we can also handle local UI transition
        } else {
            setCurrentAlert(null);
        }
    }, [externalAlert]);

    // Handle status messages from websocket
    useEffect(() => {
        if (lastMessage?.type === 'status') {
            setStatus(lastMessage.data);
        }
    }, [lastMessage]);

    if (!isActive) {
        return (
            <div className="h-full flex items-center justify-center bg-gray-800/30 rounded-2xl border border-gray-700/50 p-8">
                <div className="text-center">
                    <div className="text-6xl mb-4">📹</div>
                    <h3 className="text-gray-400 font-medium mb-2">Proctoring Inactive</h3>
                    <p className="text-gray-500 text-sm">Start a quiz to activate monitoring</p>
                </div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col space-y-4">
            {/* Proctoring Header */}
            <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-4">
                <h3 className="text-white font-bold flex items-center gap-2">
                    <span className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                    AI Proctoring Active
                </h3>
                <p className="text-gray-400 text-xs mt-1">Real-time monitoring in progress</p>
            </div>

            {/* Webcam Feed */}
            <div className="flex-shrink-0">
                <WebcamStream
                    isConnected={isConnected}
                    sendMessage={sendMessage}
                />
            </div>

            {/* Alert Display */}
            {currentAlert && (
                <div className="flex-shrink-0">
                    <AlertDisplay alert={currentAlert} />
                </div>
            )}

            {/* Status Monitor */}
            <div className="flex-1 overflow-hidden">
                <StatusMonitor
                    isConnected={isConnected}
                    status={status}
                    violationHistory={violationHistory}
                />
            </div>

            {/* Connection Error */}
            {error && (
                <div className="flex-shrink-0 bg-red-500/10 border border-red-500/30 p-3 rounded-xl">
                    <p className="text-red-300 text-xs">{error}</p>
                </div>
            )}

            {/* Audio Alert Component (invisible) */}
            <AudioAlert alert={currentAlert} language={language} />
        </div>
    );
}
