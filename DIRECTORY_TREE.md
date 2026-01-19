# NGLSC Project Directory Tree

```
NGLSC/
├── 📄 README.md                           (Original project info)
├── 📄 .gitattributes                      (Git configuration)
├── 📄 PROJECT_STRUCTURE.md                (This structure explained)
├── 📄 SETUP_GUIDE.md                      (Installation guide)
│
├── 📁 frontend/                          🎨 REACT + VITE + TAILWIND
│   ├── 📄 index.html                     (Entry HTML)
│   ├── 📄 package.json                   (npm dependencies)
│   ├── 📄 package-lock.json
│   ├── 📄 vite.config.js                 (Vite build config)
│   ├── 📄 tailwind.config.js             (Tailwind config)
│   ├── 📄 postcss.config.js              (PostCSS config)
│   │
│   ├── 📁 src/
│   │   ├── 📄 main.jsx                   (App entry point)
│   │   ├── 📄 App.jsx                    (Main routing - 14 routes)
│   │   ├── 📄 App.css                    (Global styles)
│   │   │
│   │   ├── 📁 pages/                     [9 Pages]
│   │   │   ├── 📄 Login.jsx              ✅ User login
│   │   │   ├── 📄 Register.jsx           ✅ User registration
│   │   │   ├── 📄 AdminDashboard.jsx     ✅ Admin layout wrapper
│   │   │   ├── 📄 AdminHome.jsx          ✅ Admin home page
│   │   │   ├── 📄 SmartVirtualEnvironment.jsx  ✅ Lectures & Attendance
│   │   │   ├── 📄 VideoAnalysisDashboard.jsx   ✅ Video analysis
│   │   │   ├── 📄 UserHome.jsx           ✅ Student home
│   │   │   ├── 📄 SmartAssignment.jsx    ✅ Assignments page
│   │   │   └── 📄 CognivoiceViva.jsx     ✅ Voice learning
│   │   │
│   │   ├── 📁 components/                [Reusable Components]
│   │   │   ├── 📄 AlertDisplay.jsx
│   │   │   ├── 📄 AttendanceCounter.jsx
│   │   │   ├── 📄 AudioAlert.jsx
│   │   │   ├── 📄 Footer.jsx
│   │   │   ├── 📄 Header.jsx
│   │   │   ├── 📄 Layout.jsx
│   │   │   ├── 📄 LectureGallery.jsx
│   │   │   ├── 📄 ProctoringWidget.jsx
│   │   │   ├── 📄 QuizComponent.jsx
│   │   │   ├── 📄 QuizConfigScreen.jsx
│   │   │   ├── 📄 QuizResults.jsx
│   │   │   ├── 📄 StatusMonitor.jsx
│   │   │   ├── 📄 ThemeToggle.jsx
│   │   │   ├── 📄 VoiceRecorder.jsx
│   │   │   └── 📄 WebcamStream.jsx
│   │   │
│   │   ├── 📁 context/
│   │   │   └── 📄 ThemeContext.jsx       (Dark/Light theme)
│   │   │
│   │   ├── 📁 hooks/
│   │   │   └── 📄 useWebSocket.js        (WebSocket connection)
│   │   │
│   │   └── 📁 utils/
│   │       ├── 📄 audioPlayer.js
│   │       └── 📄 (other utilities)
│   │
│   └── 📁 node_modules/                  (npm packages - auto generated)
│
├── 📁 backend/                           🔧 NODE.JS + EXPRESS + MONGODB
│   ├── 📄 server.js                      (Express server, MongoDB connection)
│   ├── 📄 package.json                   (Node.js dependencies)
│   ├── 📄 package-lock.json
│   ├── 📄 .env                           (Configuration - IMPORTANT!)
│   │
│   ├── 📁 routes/                        [API Routes]
│   │   ├── 📄 auth.js                    (register, login, verify)
│   │   └── 📄 users.js                   (GET/PUT/DELETE users)
│   │
│   ├── 📁 controllers/                   [Business Logic]
│   │   └── 📄 authController.js          (register(), login(), verify())
│   │
│   ├── 📁 middleware/                    [Express Middleware]
│   │   └── 📄 auth.js                    (JWT verification, role check)
│   │
│   ├── 📁 models/                        [Database Schemas]
│   │   └── 📄 User.js                    (MongoDB user schema)
│   │
│   ├── 📁 venv/                          (Python venv - legacy, can delete)
│   │
│   ├── 📄 requirements.txt                (Python packages - legacy)
│   │
│   └── 📄 .gitignore                     (Git ignore rules)
│
├── 📁 nglsc/                             🤖 PYTHON AI/ML - FASTAPI
│   ├── 📄 main.py                        ✅ FastAPI server, WebSocket
│   ├── 📄 config.py                      ✅ Configuration settings
│   ├── 📄 models.py                      ✅ Data models
│   ├── 📄 proctoring_engine.py           ✅ Face detection, head pose, attention
│   ├── 📄 quiz_generator.py              ✅ Quiz creation & management
│   ├── 📄 requirements.txt                ✅ Python dependencies
│   │
│   ├── 📁 models/                        (Pre-trained ML models)
│   │
│   └── 📁 lecture_gallery/               (Recorded lecture videos)
│       ├── 📹 lecture_2026-01-19_103411.webm
│       ├── 📹 lecture_2026-01-19_103419.webm
│       └── 📹 lecture_2026-01-19_103426.webm
│
└── 📁 .git/                              (Git version control)
    └── (Git internal files)
```

---

## 📊 Quick Stats

| Component | Files | Folders | Lines of Code |
|-----------|-------|---------|---------------|
| **Frontend** | 25+ | 6 | 2000+ |
| **Backend** | 8 | 4 | 800+ |
| **Python AI** | 6 | 2 | 1200+ |
| **Documentation** | 4 | 0 | 2000+ |
| **TOTAL** | 43+ | 12 | 6000+ |

---

## 🎯 Key Files by Purpose

### 🔐 Authentication
- `backend/server.js` - MongoDB + JWT setup
- `backend/routes/auth.js` - Auth endpoints
- `backend/controllers/authController.js` - Auth logic
- `backend/middleware/auth.js` - Token verification
- `frontend/src/pages/Login.jsx` - Login UI
- `frontend/src/pages/Register.jsx` - Registration UI

### 📱 User Interface
- `frontend/src/App.jsx` - Routing (14 routes)
- `frontend/src/pages/AdminDashboard.jsx` - Admin layout
- `frontend/src/pages/UserHome.jsx` - Student home
- `frontend/src/context/ThemeContext.jsx` - Theme management

### 🎓 Features
- `frontend/src/pages/SmartVirtualEnvironment.jsx` - Lectures
- `frontend/src/pages/VideoAnalysisDashboard.jsx` - Video analysis
- `frontend/src/pages/SmartAssignment.jsx` - Assignments
- `frontend/src/pages/CognivoiceViva.jsx` - Voice learning

### 🤖 AI/ML
- `nglsc/main.py` - FastAPI server
- `nglsc/proctoring_engine.py` - Face detection
- `nglsc/quiz_generator.py` - Quiz logic

---

## 🚀 Port Assignments

| Service | Port | Technology |
|---------|------|-----------|
| Frontend | 5173 | Vite Dev Server |
| Backend | 5000 | Node.js Express |
| Python AI | 8000 | FastAPI |
| MongoDB | 27017 | Database |

---

## 📦 Key Dependencies

### Frontend
- react@18.2.0
- vite@5.0.8
- tailwindcss@3.3.6
- react-router-dom@7.10.1

### Backend
- express@4.18.2
- mongodb@6.3.0
- jsonwebtoken@9.1.2
- bcryptjs@2.4.3

### Python
- fastapi
- uvicorn
- mediapipe
- opencv-python
- numpy

---

**This structure provides a clean separation between:**
- 🎨 Frontend (React/UI)
- 🔧 Backend (Node.js/API)
- 🤖 AI/ML (Python/Analytics)

**All components can run independently or together!**
