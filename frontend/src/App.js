import React, { useState, useEffect } from 'react';
import './App.css';
import MCQQuestion from './components/MCQQuestion';
import DiagramQuestion from './components/DiagramQuestion';
import ScoreSummary from './components/ScoreSummary';
import { fetchMCQQuestions, submitMCQAnswer, fetchERQuestion, submitDiagramAnswer } from './services/api';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [mcqQuestions, setMcqQuestions] = useState([]);
  const [erQuestion, setErQuestion] = useState(null);
  const [flowchartQuestion, setFlowchartQuestion] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [scores, setScores] = useState({
    mcq: [],
    er: null,
    flowchart: null,
    totalScore: 0,
    totalQuestions: 12
  });

  useEffect(() => {
    loadQuestions();
  }, []);

  const loadQuestions = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch 10 MCQ questions
      const mcqData = await fetchMCQQuestions(10);
      setMcqQuestions(mcqData.questions.map(q => ({
        ...q,
        answered: false,
        result: null
      })));

      // Fetch 1 ER diagram question
      const erData = await fetchERQuestion();
      setErQuestion({
        ...erData,
        type: 'er',
        answered: false,
        result: null
      });

      // Fetch 1 Flowchart question
      const flowchartData = await fetchERQuestion();
      setFlowchartQuestion({
        ...flowchartData,
        type: 'flowchart',
        answered: false,
        result: null
      });

      setLoading(false);
    } catch (err) {
      setError(err.message || 'Failed to load questions');
      setLoading(false);
    }
  };

  const handleMCQAnswer = async (questionId, selectedAnswer) => {
    try {
      const response = await submitMCQAnswer([{
        id: questionId,
        selected: selectedAnswer
      }]);

      if (response.results && response.results.length > 0) {
        const result = response.results[0];
        
        // Update the question with result
        setMcqQuestions(prev => prev.map(q => 
          q.id === questionId 
            ? { ...q, answered: true, result, selectedAnswer }
            : q
        ));

        // Update scores
        setScores(prev => {
          const newMcqScores = prev.mcq.filter(s => s.id !== questionId);
          newMcqScores.push({
            id: questionId,
            isCorrect: result.is_correct,
            score: result.is_correct ? 1 : 0
          });
          
          const newTotalScore = 
            newMcqScores.reduce((sum, s) => sum + s.score, 0) +
            (prev.er?.score || 0) +
            (prev.flowchart?.score || 0);

          return {
            ...prev,
            mcq: newMcqScores,
            totalScore: newTotalScore
          };
        });
      }
    } catch (err) {
      console.error('Error submitting MCQ answer:', err);
    }
  };

  const handleDiagramAnswer = async (questionType, questionId, imageFile) => {
    try {
      const response = await submitDiagramAnswer(questionId, imageFile);
      
      const result = {
        is_correct: response.is_correct,
        confidence: response.confidence,
        predicted_class: response.predicted_class
      };

      if (questionType === 'er') {
        setErQuestion(prev => ({
          ...prev,
          answered: true,
          result
        }));
      } else {
        setFlowchartQuestion(prev => ({
          ...prev,
          answered: true,
          result
        }));
      }

      // Update scores
      setScores(prev => {
        const newScore = {
          type: questionType,
          isCorrect: result.is_correct,
          score: result.is_correct ? 1 : 0
        };

        const newTotalScore = 
          prev.mcq.reduce((sum, s) => sum + s.score, 0) +
          (questionType === 'er' ? newScore.score : (prev.er?.score || 0)) +
          (questionType === 'flowchart' ? newScore.score : (prev.flowchart?.score || 0));

        return {
          ...prev,
          [questionType]: newScore,
          totalScore: newTotalScore
        };
      });
    } catch (err) {
      console.error('Error submitting diagram answer:', err);
    }
  };

  if (loading) {
    return (
      <div className="app-container">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading questions...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app-container">
        <div className="error-container">
          <h2>Error</h2>
          <p>{error}</p>
          <button onClick={loadQuestions} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <div className="app-header">
        <h1>📚 Assignment System</h1>
        <p className="subtitle">Answer the questions below. You'll receive immediate feedback!</p>
      </div>

      <ScoreSummary scores={scores} />

      <div className="questions-container">
        {/* MCQ Questions Section */}
        <section className="question-section">
          <h2 className="section-title">Multiple Choice Questions (10 questions)</h2>
          <div className="mcq-grid">
            {mcqQuestions.map((question, index) => (
              <MCQQuestion
                key={question.id}
                question={question}
                questionNumber={index + 1}
                onAnswer={handleMCQAnswer}
              />
            ))}
          </div>
        </section>

        {/* ER Diagram Section */}
        <section className="question-section">
          <h2 className="section-title">ER Diagram Question</h2>
          {erQuestion && (
            <DiagramQuestion
              question={erQuestion}
              questionNumber={11}
              onAnswer={handleDiagramAnswer}
            />
          )}
        </section>

        {/* Flowchart Section */}
        <section className="question-section">
          <h2 className="section-title">Flowchart Question</h2>
          {flowchartQuestion && (
            <DiagramQuestion
              question={flowchartQuestion}
              questionNumber={12}
              onAnswer={handleDiagramAnswer}
            />
          )}
        </section>
      </div>
    </div>
  );
}

export default App;

