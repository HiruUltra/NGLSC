import React, { useState } from 'react';
import './MCQQuestion.css';

const MCQQuestion = ({ question, questionNumber, onAnswer }) => {
  const [selectedOption, setSelectedOption] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleOptionSelect = async (option) => {
    if (question.answered || isSubmitting) return;

    setSelectedOption(option);
    setIsSubmitting(true);

    try {
      await onAnswer(question.id, option);
    } catch (error) {
      console.error('Error submitting answer:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const getResultClass = () => {
    if (!question.answered || !question.result) return '';
    return question.result.is_correct ? 'correct' : 'incorrect';
  };

  const getResultIcon = () => {
    if (!question.answered || !question.result) return null;
    return question.result.is_correct ? '✓' : '✗';
  };

  return (
    <div className={`mcq-question-card ${getResultClass()}`}>
      <div className="question-header">
        <span className="question-number">Q{questionNumber}</span>
        {question.answered && question.result && (
          <span className={`result-badge ${question.result.is_correct ? 'correct' : 'incorrect'}`}>
            {getResultIcon()} {question.result.is_correct ? 'Correct' : 'Incorrect'}
            {question.result.is_correct && <span className="score-badge">+1</span>}
          </span>
        )}
      </div>

      <p className="question-text">{question.question}</p>

      <div className="options-container">
        {question.options.map((option, index) => {
          const isSelected = selectedOption === option;
          const isCorrect = question.answered && question.result && question.result.correct_answer === option;
          const isWrong = question.answered && question.result && !question.result.is_correct && isSelected;

          return (
            <button
              key={index}
              className={`option-button ${
                isSelected ? 'selected' : ''
              } ${
                isCorrect ? 'correct-answer' : ''
              } ${
                isWrong ? 'wrong-answer' : ''
              }`}
              onClick={() => handleOptionSelect(option)}
              disabled={question.answered || isSubmitting}
            >
              <span className="option-label">{String.fromCharCode(65 + index)}.</span>
              <span className="option-text">{option}</span>
              {isCorrect && <span className="option-icon">✓</span>}
              {isWrong && <span className="option-icon">✗</span>}
            </button>
          );
        })}
      </div>

      {question.answered && question.result && !question.result.is_correct && (
        <div className="feedback-message">
          <strong>Correct Answer:</strong> {question.result.correct_answer}
        </div>
      )}

      {isSubmitting && (
        <div className="submitting-overlay">
          <span>Checking answer...</span>
        </div>
      )}
    </div>
  );
};

export default MCQQuestion;

