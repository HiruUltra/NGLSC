# Backend - MCQ and Diagram Assignment API

Flask backend API for MCQ questions and ER/Flowchart diagram grading using ML models.

## Features

- **MCQ Questions API**: Serve random MCQ questions and grade answers using ML models
- **ER/Flowchart Diagram API**: Serve diagram questions and grade uploaded images using computer vision models
- **Model Training**: Optional endpoints for training new models
- **CORS Enabled**: Ready for frontend integration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables (optional):
```bash
# MCQ Configuration
export MCQ_CSV_PATH=mcq.csv
export MCQ_MODEL_PATH=outputs/latest_mcq/best_model.joblib

# ER/Flowchart Configuration
export ER_RUN_ROOT=outputs/20260104_001541
export ER_BEST_MODEL_PATH=outputs/20260104_001541/BEST/best_model.pth
export ER_BEST_BACKBONE_PATH=outputs/20260104_001541/BEST/best_backbone.txt
export ER_QUESTIONS_PATH=data/er_questions.xlsx

# Optional: Enable training endpoints
export ENABLE_TRAIN_ENDPOINTS=1
```

3. Run the server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### MCQ Endpoints

- `GET /quiz?n=10` - Get N random MCQ questions
- `POST /submit` - Submit MCQ answers and get graded results

### ER/Flowchart Endpoints

- `GET /er/question` - Get a random ER/Flowchart question
- `POST /er/submit` - Submit a diagram image for grading

### Debug Endpoints

- `GET /debug/er` - Check ER model status and configuration

### Training Endpoints (Optional)

- `POST /admin/train/mcq` - Train a new MCQ model
- `POST /admin/train/er` - Train a new ER/Flowchart model

## Project Structure

```
backend/
  ├── app.py                 # Main Flask application
  ├── train_mcq_model.py     # MCQ model training script
  ├── train_er_model.py      # ER/Flowchart model training script
  ├── mcq.csv                # MCQ question bank
  ├── data/                  # Training data
  │   ├── er_images/         # ER/Flowchart images by class
  │   └── er_questions.xlsx  # Question mapping
  ├── outputs/               # Trained models and results
  └── user_responses.csv     # Logged user responses
```

## File Structure Requirements

- `mcq.csv`: Must contain `question` and `correct_answer` columns
- `data/er_images/`: Folders named by class (e.g., ER1, ER2, FlowChart1, etc.)
- `outputs/`: Contains trained models and training results



