import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../components/Header';

export default function StudentProfile() {
  const navigate = useNavigate();
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');
  
  const [formData, setFormData] = useState(() => {
    const user = localStorage.getItem('user');
    return user ? JSON.parse(user) : {
      name: '',
      email: '',
      phoneNumber: '',
      department: '',
      rollNumber: '',
      semester: ''
    };
  });

  const [editData, setEditData] = useState(formData);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setEditData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSaveChanges = async () => {
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const token = localStorage.getItem('token');
      
      const response = await fetch('http://localhost:5000/api/users/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(editData)
      });

      if (response.ok) {
        const updatedUser = await response.json();
        localStorage.setItem('user', JSON.stringify(updatedUser.user));
        setFormData(updatedUser.user);
        setIsEditing(false);
        setSuccess('✅ Profile updated successfully!');
        setTimeout(() => setSuccess(''), 3000);
      } else {
        setError('❌ Failed to update profile');
      }
    } catch (err) {
      setError('❌ Error updating profile: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Header />
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black">
        <div className="max-w-4xl mx-auto px-6 py-12">
          {/* Header Section */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
            <div className="flex items-center gap-6 mb-6">
              <div className="w-24 h-24 rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 text-white text-4xl font-bold flex items-center justify-center">
                {formData.name ? formData.name.charAt(0).toUpperCase() : '👤'}
              </div>
              <div>
                <h1 className="text-4xl font-bold text-gray-900 dark:text-white">{formData.name || 'Student'}</h1>
                <p className="text-gray-600 dark:text-gray-400">{formData.email}</p>
                <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">Roll Number: {formData.rollNumber || 'N/A'}</p>
              </div>
            </div>

            {success && (
              <div className="mb-4 p-4 bg-green-100 dark:bg-green-900/30 border border-green-400 rounded-lg text-green-700 dark:text-green-400">
                {success}
              </div>
            )}

            {error && (
              <div className="mb-4 p-4 bg-red-100 dark:bg-red-900/30 border border-red-400 rounded-lg text-red-700 dark:text-red-400">
                {error}
              </div>
            )}
          </div>

          {/* Profile Information Section */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Profile Information</h2>
              <button
                onClick={() => {
                  if (isEditing) {
                    setEditData(formData);
                  }
                  setIsEditing(!isEditing);
                }}
                className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg font-semibold transition-colors"
              >
                {isEditing ? '❌ Cancel' : '✏️ Edit Profile'}
              </button>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              {/* Personal Information */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Personal Information</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Full Name</label>
                    {isEditing ? (
                      <input
                        type="text"
                        name="name"
                        value={editData.name}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-cyan-500 outline-none"
                      />
                    ) : (
                      <p className="px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg text-gray-900 dark:text-white">
                        {formData.name || 'N/A'}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Email</label>
                    {isEditing ? (
                      <input
                        type="email"
                        name="email"
                        value={editData.email}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-cyan-500 outline-none"
                      />
                    ) : (
                      <p className="px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg text-gray-900 dark:text-white">
                        {formData.email || 'N/A'}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Phone Number</label>
                    {isEditing ? (
                      <input
                        type="tel"
                        name="phoneNumber"
                        value={editData.phoneNumber || ''}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-cyan-500 outline-none"
                      />
                    ) : (
                      <p className="px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg text-gray-900 dark:text-white">
                        {formData.phoneNumber || 'N/A'}
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {/* Academic Information */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Academic Information</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Roll Number</label>
                    {isEditing ? (
                      <input
                        type="text"
                        name="rollNumber"
                        value={editData.rollNumber || ''}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-cyan-500 outline-none"
                      />
                    ) : (
                      <p className="px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg text-gray-900 dark:text-white">
                        {formData.rollNumber || 'N/A'}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Department</label>
                    {isEditing ? (
                      <input
                        type="text"
                        name="department"
                        value={editData.department || ''}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-cyan-500 outline-none"
                      />
                    ) : (
                      <p className="px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg text-gray-900 dark:text-white">
                        {formData.department || 'N/A'}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Semester</label>
                    {isEditing ? (
                      <input
                        type="text"
                        name="semester"
                        value={editData.semester || ''}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-cyan-500 outline-none"
                      />
                    ) : (
                      <p className="px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg text-gray-900 dark:text-white">
                        {formData.semester || 'N/A'}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {isEditing && (
              <div className="mt-8 flex gap-4">
                <button
                  onClick={handleSaveChanges}
                  disabled={loading}
                  className="px-6 py-3 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white rounded-lg font-semibold transition-colors"
                >
                  {loading ? '⏳ Saving...' : '💾 Save Changes'}
                </button>
                <button
                  onClick={() => {
                    setEditData(formData);
                    setIsEditing(false);
                  }}
                  className="px-6 py-3 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-semibold transition-colors"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>

          {/* Additional Stats */}
          <div className="mt-8 grid md:grid-cols-3 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <div className="text-3xl mb-2">📊</div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Average Score</h3>
              <p className="text-3xl font-bold text-cyan-500">85%</p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <div className="text-3xl mb-2">📋</div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Assignments Completed</h3>
              <p className="text-3xl font-bold text-blue-500">12</p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <div className="text-3xl mb-2">🎯</div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Quizzes Taken</h3>
              <p className="text-3xl font-bold text-purple-500">8</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
