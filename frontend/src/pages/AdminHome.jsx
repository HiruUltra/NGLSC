export default function AdminHome() {
  const stats = [
    { label: 'Total Students', value: '254', icon: '👨‍🎓', color: 'from-blue-500 to-blue-600' },
    { label: 'Active Lectures', value: '12', icon: '📚', color: 'from-green-500 to-green-600' },
    { label: 'Exams Today', value: '8', icon: '📝', color: 'from-purple-500 to-purple-600' },
    { label: 'Avg Score', value: '78%', icon: '📊', color: 'from-orange-500 to-orange-600' }
  ];

  const recentActivity = [
    { id: 1, action: 'Quiz submitted by John Doe', time: '5 mins ago' },
    { id: 2, action: 'Lecture uploaded: Advanced Mathematics', time: '1 hour ago' },
    { id: 3, action: 'Student registered: Sarah Smith', time: '2 hours ago' },
    { id: 4, action: 'Assignment deadline: Physics Project', time: '3 hours ago' }
  ];

  return (
    <div className="p-8">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, idx) => (
          <div key={idx} className={`bg-gradient-to-br ${stat.color} rounded-xl p-6 text-white shadow-lg`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-100 text-sm font-medium">{stat.label}</p>
                <p className="text-4xl font-bold mt-2">{stat.value}</p>
              </div>
              <span className="text-5xl opacity-30">{stat.icon}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h2 className="text-xl font-bold text-white mb-6">Recent Activity</h2>
          <div className="space-y-4">
            {recentActivity.map(activity => (
              <div key={activity.id} className="flex items-start gap-4 pb-4 border-b border-gray-700 last:border-0">
                <div className="w-3 h-3 bg-blue-500 rounded-full mt-1.5 flex-shrink-0"></div>
                <div className="flex-1">
                  <p className="text-white font-medium">{activity.action}</p>
                  <p className="text-gray-400 text-sm">{activity.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h2 className="text-xl font-bold text-white mb-6">Quick Actions</h2>
          <div className="space-y-3">
            <button className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded-lg font-medium transition">
              Create Quiz
            </button>
            <button className="w-full bg-green-600 hover:bg-green-700 text-white py-2 rounded-lg font-medium transition">
              Upload Lecture
            </button>
            <button className="w-full bg-purple-600 hover:bg-purple-700 text-white py-2 rounded-lg font-medium transition">
              Create Assignment
            </button>
            <button className="w-full bg-orange-600 hover:bg-orange-700 text-white py-2 rounded-lg font-medium transition">
              View Reports
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
