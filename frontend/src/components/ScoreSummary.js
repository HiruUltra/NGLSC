import React from 'react';
import './ScoreSummary.css';

const ScoreSummary = ({ scores }) => {
  const mcqAnswered = scores.mcq.length;
  const mcqCorrect = scores.mcq.filter(s => s.isCorrect).length;
  const totalAnswered = mcqAnswered + (scores.er ? 1 : 0) + (scores.flowchart ? 1 : 0);
  const totalCorrect = mcqCorrect + (scores.er?.isCorrect ? 1 : 0) + (scores.flowchart?.isCorrect ? 1 : 0);

  return (
    <div className="score-summary">
      <div className="score-card">
        <div className="score-label">Total Score</div>
        <div className="score-value">
          {scores.totalScore} / {scores.totalQuestions}
        </div>
        <div className="score-percentage">
          {scores.totalQuestions > 0 
            ? Math.round((scores.totalScore / scores.totalQuestions) * 100) 
            : 0}%
        </div>
      </div>

      <div className="score-breakdown">
        <div className="breakdown-item">
          <span className="breakdown-label">MCQ Questions</span>
          <span className="breakdown-value">
            {mcqCorrect} / {mcqAnswered} answered
          </span>
        </div>
        <div className="breakdown-item">
          <span className="breakdown-label">ER Diagram</span>
          <span className="breakdown-value">
            {scores.er ? (scores.er.isCorrect ? '✓ Correct' : '✗ Incorrect') : 'Not answered'}
          </span>
        </div>
        <div className="breakdown-item">
          <span className="breakdown-label">Flowchart</span>
          <span className="breakdown-value">
            {scores.flowchart ? (scores.flowchart.isCorrect ? '✓ Correct' : '✗ Incorrect') : 'Not answered'}
          </span>
        </div>
      </div>

      <div className="progress-bar-container">
        <div className="progress-bar">
          <div 
            className="progress-fill"
            style={{ width: `${(scores.totalScore / scores.totalQuestions) * 100}%` }}
          ></div>
        </div>
        <div className="progress-text">
          {totalAnswered} of {scores.totalQuestions} questions answered
        </div>
      </div>
    </div>
  );
};

export default ScoreSummary;

