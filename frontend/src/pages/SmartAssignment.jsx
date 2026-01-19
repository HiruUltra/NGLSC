import Header from '../components/Header';

export default function SmartAssignment() {
  const assignments = [
    { id: 1, title: 'Physics Project', dueDate: '2024-02-15', status: 'In Progress' },
    { id: 2, title: 'Math Assignment 5', dueDate: '2024-02-20', status: 'Not Started' },
    { id: 3, title: 'Chemistry Lab Report', dueDate: '2024-02-10', status: 'Submitted' }
  ];

  return (
    <>
      <Header />
      <div className="min-h-screen bg-gray-900 p-8">
        <h1 className="text-4xl font-bold text-white mb-2">Smart Monthly Assignment</h1>
        <p className="text-gray-400 mb-8">Complete your assignments with AI-powered feedback</p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          {assignments.map(assignment => (
            <div key={assignment.id} className="bg-gray-800 rounded-xl p-6 border border-gray-700 hover:border-gray-600 transition">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-white">{assignment.title}</h3>
                  <p className="text-gray-400 text-sm mt-2">Due: {assignment.dueDate}</p>
                </div>
                <span className={`px-4 py-2 rounded-full text-sm font-semibold ${
                  assignment.status === 'Submitted' ? 'bg-green-500/20 text-green-400' :
                  assignment.status === 'In Progress' ? 'bg-blue-500/20 text-blue-400' :
                  'bg-gray-500/20 text-gray-400'
                }`}>
                  {assignment.status}
                </span>
              </div>
              <button className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition">
                {assignment.status === 'Submitted' ? 'View Feedback' : 'Open Assignment'}
              </button>
            </div>
          ))}
        </div>

        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 h-fit">
          <h3 className="text-xl font-bold text-white mb-4">📌 Tips</h3>
          <ul className="space-y-3 text-gray-300 text-sm">
            <li>• Start early to get AI feedback</li>
            <li>• Use the suggestion tool</li>
            <li>• Submit before deadline</li>
            <li>• Review instructor comments</li>
          </ul>
        </div>
        </div>
      </div>
    </>
  );
}
