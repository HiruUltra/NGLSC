import { Link } from 'react-router-dom';
import Header from '../components/Header';

export default function UserHome() {
  const user = JSON.parse(localStorage.getItem('user') || '{}');

  const features = [
    {
      id: 1,
      title: 'Quiz System',
      description: 'Take interactive quizzes with real-time proctoring and instant feedback',
      icon: '📝',
      color: 'from-blue-500 to-blue-600',
      path: '/quiz'
    },
    {
      id: 2,
      title: 'Smart Monthly Assignment',
      description: 'Complete monthly assignments with AI-powered suggestions and feedback',
      icon: '📚',
      color: 'from-green-500 to-green-600',
      path: '/assignments'
    },
    {
      id: 3,
      title: 'Cognivoice Viva Assistant',
      description: 'Practice with our AI voice assistant for oral exams and presentations',
      icon: '🎤',
      color: 'from-purple-500 to-purple-600',
      path: '/cognivoice'
    }
  ];

  return (
    <>
      <Header />
      <div className="min-h-screen bg-gray-900">
        {/* Header */}
        <header className="bg-gray-800 border-b border-gray-700 px-8 py-6">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-4xl font-bold text-white">Welcome, {user.name}! 👋</h1>
          <p className="text-gray-400 mt-2">Ready to excel in your learning journey?</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-8 py-12">
        {/* Featured Courses/Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {[
            { label: 'Courses Enrolled', value: '5', icon: '📖' },
            { label: 'Average Score', value: '85%', icon: '⭐' },
            { label: 'Assignments Due', value: '3', icon: '📋' }
          ].map((stat, idx) => (
            <div key={idx} className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700">
              <p className="text-gray-400 text-sm">{stat.label}</p>
              <p className="text-4xl font-bold text-white mt-2">{stat.value}</p>
              <span className="text-3xl mt-2 block">{stat.icon}</span>
            </div>
          ))}
        </div>

        {/* Feature Cards */}
        <section>
          <h2 className="text-3xl font-bold text-white mb-8">Learning Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map(feature => (
              <Link
                key={feature.id}
                to={feature.path}
                className="group"
              >
                <div className={`bg-gradient-to-br ${feature.color} rounded-xl p-8 text-white shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 h-full`}>
                  <div className="text-6xl mb-4 group-hover:scale-110 transition duration-300">{feature.icon}</div>
                  <h3 className="text-2xl font-bold mb-2">{feature.title}</h3>
                  <p className="text-white/90 mb-6">{feature.description}</p>
                  <div className="flex items-center gap-2 text-white font-semibold group-hover:gap-3 transition">
                    Start Now <span>→</span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Home Navigation */}
        <section className="mt-16 pt-8 border-t border-gray-700">
          <h2 className="text-3xl font-bold text-white mb-8">More</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Link to="/home" className="bg-gray-800 hover:bg-gray-700 rounded-xl p-6 border border-gray-700 transition">
              <h3 className="text-xl font-bold text-white mb-2">🏠 Home Overview</h3>
              <p className="text-gray-400">View your dashboard and quick stats</p>
            </Link>
            <Link to="/profile" className="bg-gray-800 hover:bg-gray-700 rounded-xl p-6 border border-gray-700 transition">
              <h3 className="text-xl font-bold text-white mb-2">👤 My Profile</h3>
              <p className="text-gray-400">Manage your account and preferences</p>
            </Link>
          </div>
        </section>
      </main>
      </div>
    </>
  );
}
