import { useRef, useEffect, useState } from 'react';

/**
 * WebcamStream Component
 * Captures webcam feed and sends frames to backend via WebSocket
 */
export default function WebcamStream({ isConnected, sendMessage }) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const streamRef = useRef(null);
    const intervalRef = useRef(null);

    const [cameraStatus, setCameraStatus] = useState('initializing');
    const [error, setError] = useState(null);

    // Initialize webcam
    useEffect(() => {
        const startCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        frameRate: { ideal: 30 }
                    },
                    audio: false
                });

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    streamRef.current = stream;
                    setCameraStatus('active');
                    setError(null);
                }
            } catch (err) {
                console.error('Camera access error:', err);
                setCameraStatus('error');

                if (err.name === 'NotAllowedError') {
                    setError('Camera permission denied. Please allow camera access.');
                } else if (err.name === 'NotFoundError') {
                    setError('No camera found. Please connect a camera.');
                } else {
                    setError(`Camera error: ${err.message}`);
                }
            }
        };

        startCamera();

        return () => {
            // Cleanup camera on unmount
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    // Send frames at 5 FPS when connected
    useEffect(() => {
        if (!isConnected || cameraStatus !== 'active') {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
            return;
        }

        const captureAndSend = () => {
            if (!videoRef.current || !canvasRef.current) return;

            const video = videoRef.current;
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');

            // Set canvas size to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            // Draw current video frame to canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert to JPEG and encode as base64
            canvas.toBlob((blob) => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64data = reader.result.split(',')[1];

                    // Send to backend
                    sendMessage({
                        frame: base64data,
                        timestamp: new Date().toISOString()
                    });
                };
                reader.readAsDataURL(blob);
            }, 'image/jpeg', 0.8); // JPEG quality 80%
        };

        // Capture at 5 FPS (every 200ms)
        intervalRef.current = setInterval(captureAndSend, 200);

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [isConnected, cameraStatus, sendMessage]);

    return (
        <div className="webcam-container">
            <div className="relative rounded-2xl overflow-hidden bg-gray-900 shadow-2xl">
                {/* Video feed */}
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-auto"
                />

                {/* Hidden canvas for frame capture */}
                <canvas ref={canvasRef} className="hidden" />

                {/* Status overlay */}
                <div className="absolute top-4 left-4 flex items-center gap-2">
                    <div className={`status-indicator ${cameraStatus === 'active' ? 'active' : 'inactive'}`} />
                    <span className="text-white text-sm font-medium bg-black/50 px-3 py-1 rounded-full backdrop-blur-sm">
                        {cameraStatus === 'active' ? 'Camera Active' : 'Camera Inactive'}
                    </span>
                </div>

                {/* Connection status */}
                <div className="absolute top-4 right-4">
                    <span className={`text-xs font-medium px-3 py-1 rounded-full backdrop-blur-sm ${isConnected
                            ? 'bg-green-500/20 text-green-300 border border-green-500/30'
                            : 'bg-red-500/20 text-red-300 border border-red-500/30'
                        }`}>
                        {isConnected ? '● Connected' : '○ Disconnected'}
                    </span>
                </div>

                {/* Error overlay */}
                {error && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/80 backdrop-blur-sm">
                        <div className="max-w-md p-6 bg-red-500/10 border border-red-500/30 rounded-xl text-center">
                            <div className="text-4xl mb-3">⚠️</div>
                            <h3 className="text-red-300 font-semibold mb-2">Camera Error</h3>
                            <p className="text-red-200 text-sm">{error}</p>
                        </div>
                    </div>
                )}
            </div>

            <style jsx>{`
        .status-indicator {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          background-color: #ef4444;
        }
        
        .status-indicator.active {
          background-color: #10b981;
          box-shadow: 0 0 10px #10b981;
          animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
      `}</style>
        </div>
    );
}
