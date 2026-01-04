import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchMCQQuestions = async (count = 10) => {
  try {
    const response = await api.get(`/quiz?n=${count}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to fetch MCQ questions');
  }
};

export const submitMCQAnswer = async (answers) => {
  try {
    const response = await api.post('/submit', {
      answers: answers,
      user_id: 'student_' + Date.now() // You can replace this with actual user ID
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to submit MCQ answer');
  }
};

export const fetchERQuestion = async () => {
  try {
    const response = await api.get('/er/question');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to fetch diagram question');
  }
};

export const processDiagram = async (imageFile) => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await api.post('/er/convert', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data; // { svg: "<svg>...</svg>" }
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to process diagram');
  }
};

export const submitDiagramAnswer = async (questionId, imageFile) => {
  try {
    const formData = new FormData();
    formData.append('question_id', questionId);
    formData.append('image', imageFile);

    const response = await api.post('/er/submit', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to submit diagram answer');
  }
};

