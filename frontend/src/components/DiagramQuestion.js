import React, { useState, useRef } from 'react';
import './DiagramQuestion.css';
import { processDiagram } from '../services/api';

const DiagramQuestion = ({ question, questionNumber, onAnswer }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [processedSVG, setProcessedSVG] = useState(null);
  const [processingSVG, setProcessingSVG] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const fileInputRef = useRef(null);

  const fetchProcessedSVG = async (file) => {
    setProcessingSVG(true);
    setProcessedSVG(null);
    try {
      const resp = await processDiagram(file);
      if (resp && resp.svg) {
        setProcessedSVG(resp.svg);
      }
    } catch (err) {
      console.error('Diagram processing failed:', err);
      setProcessedSVG(null);
    } finally {
      setProcessingSVG(false);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);

      // Start processing in background
      fetchProcessedSVG(file);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile || question.answered || isSubmitting) return;

    setIsSubmitting(true);
    try {
      await onAnswer(question.type, question.question_id, selectedFile);
    } catch (error) {
      console.error('Error submitting diagram:', error);
      alert('Failed to submit diagram. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getResultClass = () => {
    if (!question.answered || !question.result) return '';
    return question.result.is_correct ? 'correct' : 'incorrect';
  };

  return (
    <div className={`diagram-question-card ${getResultClass()}`}>
      <div className="question-header">
        <span className="question-number">Q{questionNumber}</span>
        {question.answered && question.result && (
          <span className={`result-badge ${question.result.is_correct ? 'correct' : 'incorrect'}`}>
            {question.result.is_correct ? '✓ Correct' : '✗ Incorrect'}
            {question.result.is_correct && <span className="score-badge">+1</span>}
          </span>
        )}
      </div>

      <p className="question-text">{question.question_text}</p>

      {!question.answered ? (
        <div className="diagram-upload-section">
          <div className="upload-area">
            {preview ? (
              <div className="preview-container">
                <img src={preview} alt="Preview" className="preview-image" />
                <button onClick={handleReset} className="remove-button">
                  Remove Image
                </button>

                {processingSVG && <div className="processing-indicator">Processing…</div>}

                {processedSVG && (
                  <div className="processed-preview">
                    <p className="processed-label">Cleaned preview:</p>
                    <div className="svg-preview" dangerouslySetInnerHTML={{ __html: processedSVG }} />
                  </div>
                )}
              </div>
            ) : (
              <div className="upload-placeholder">
                <div className="upload-icon">📤</div>
                <p>Click to upload or drag and drop</p>
                <p className="upload-hint">PNG, JPG, JPEG up to 10MB</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="file-input"
                />
              </div>
            )}
          </div>

          {selectedFile && (
            <div className="submit-section">
              <p className="file-name">Selected: {selectedFile.name}</p>
              <button
                onClick={handleSubmit}
                disabled={isSubmitting}
                className="submit-button"
              >
                {isSubmitting ? 'Submitting...' : 'Submit Answer'}
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="result-section">
          {preview && (
            <div className="submitted-image">
              <img src={preview} alt="Submitted" className="result-image" />
            </div>
          )}
          <div className="result-details">
            <p className={`result-status ${question.result.is_correct ? 'correct' : 'incorrect'}`}>
              {question.result.is_correct 
                ? '✓ Your diagram is correct!' 
                : '✗ Your diagram does not match the expected answer.'}
            </p>
            {question.result.confidence !== null && (
              <p className="confidence-score">
                Confidence: {(question.result.confidence * 100).toFixed(1)}%
              </p>
            )}
            {question.result.predicted_class && (
              <p className="predicted-class">
                Detected: {question.result.predicted_class}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default DiagramQuestion;

