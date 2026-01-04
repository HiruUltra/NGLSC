# Frontend - React Assignment System

Modern React frontend for the MCQ and Diagram Assignment System.

## Features

- **10 MCQ Questions**: Students answer multiple choice questions one by one with immediate feedback
- **1 ER Diagram Question**: Students upload an ER diagram image and receive instant grading
- **1 Flowchart Question**: Students upload a flowchart image and receive instant grading
- **Real-time Scoring**: Live score updates as students answer questions
- **Beautiful UI**: Modern, responsive design with smooth animations

## Setup

1. Install dependencies:
```bash
npm install
```

2. Configure API URL (optional):
   - Create a `.env` file in the frontend directory
   - Add: `REACT_APP_API_URL=http://localhost:5000`
   - Default is `http://localhost:5000`

3. Start the development server:
```bash
npm start
```

The app will open at `http://localhost:3000`

## Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build/` folder.

## Project Structure

```
frontend/
  ├── public/
  │   └── index.html
  ├── src/
  │   ├── components/
  │   │   ├── MCQQuestion.js       # MCQ question component
  │   │   ├── MCQQuestion.css
  │   │   ├── DiagramQuestion.js   # ER/Flowchart question component
  │   │   ├── DiagramQuestion.css
  │   │   ├── ScoreSummary.js      # Score display component
  │   │   └── ScoreSummary.css
  │   ├── services/
  │   │   └── api.js               # API service functions
  │   ├── App.js                   # Main app component
  │   ├── App.css
  │   ├── index.js                 # Entry point
  │   └── index.css
  ├── package.json
  └── README.md
```

## Backend Requirements

Make sure your Flask backend is running on `http://localhost:5000` (or update the API URL in `.env`).

The backend should have these endpoints:
- `GET /quiz?n=10` - Get MCQ questions
- `POST /submit` - Submit MCQ answers
- `GET /er/question` - Get ER/Flowchart question
- `POST /er/submit` - Submit diagram image
