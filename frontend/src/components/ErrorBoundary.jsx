import React from 'react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error("3D Context Error:", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="flex flex-col items-center justify-center h-full bg-gray-900 text-white p-8 text-center">
                    <div className="text-6xl mb-6">⚠️</div>
                    <h2 className="text-2xl font-bold mb-4 text-red-400">3D Rendering Error</h2>
                    <p className="text-gray-400 max-w-md mb-8">
                        Your browser or device might not support WebGL2 properly, or there was a conflict in the rendering engine.
                    </p>
                    <button
                        onClick={() => window.location.reload()}
                        className="px-8 py-3 bg-cyan-600 hover:bg-cyan-700 rounded-xl font-bold transition-all"
                    >
                        🔄 Refresh Page
                    </button>
                    <div className="mt-8 p-4 bg-black/50 rounded-lg text-xs font-mono text-gray-500 overflow-auto max-w-full">
                        {this.state.error?.toString()}
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
