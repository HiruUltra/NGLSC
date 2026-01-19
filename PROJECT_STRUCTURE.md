# 🎯 NGLSC Project Structure

## Folder Organization

```
nglsc-project/
│
├── frontend/                 🎨 React (Vite + Tailwind)
│   ├── src/
│   │   ├── pages/           [9 pages: Login, Register, Admin, Student]
│   │   ├── components/      [Reusable React components]
│   │   ├── context/         [Theme context provider]
│   │   ├── hooks/           [Custom hooks]
│   │   ├── App.jsx          [Main routing with ProtectedRoute]
│   │   ├── App.css
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── index.html
│
├── backend/                 🔧 Node.js + Express + MongoDB
│   ├── server.js            [Express server, MongoDB connection]
│   ├── package.json         [Dependencies: express, mongodb, jwt, bcrypt]
│   ├── .env                 [MongoDB URI, JWT secrets, config]
│   │
│   ├── routes/
│   │   ├── auth.js          [POST /register, /login, GET /verify]
│   │   └── users.js         [GET /me, /all, PUT /:id, DELETE /:id]
│   │
│   ├── controllers/
│   │   └── authController.js [register(), login(), verifyEmail()]
│   │
│   ├── middleware/
│   │   └── auth.js          [verifyToken(), requireRole()]
│   │
│   ├── models/
│   │   └── User.js          [MongoDB user schema]
│   │
│   ├── venv/                [Python virtual environment (legacy)]
│   └── requirements.txt      [Python dependencies (legacy)]
│
└── nglsc/                   🤖 Python AI/ML Services
    ├── main.py              [FastAPI server, WebSocket]
    ├── config.py            [Configuration settings]
    ├── models.py            [Data models]
    ├── proctoring_engine.py [Face detection, head pose, attention]
    ├── quiz_generator.py    [Quiz creation & management]
    ├── requirements.txt     [Python dependencies: FastAPI, MediaPipe, OpenCV]
    │
    ├── models/              [Pre-trained ML models]
    └── lecture_gallery/     [Recorded lecture videos]
```

---

## 📊 File Inventory

### Frontend Files (React)
- ✅ `App.jsx` - 14 routes with protected access
- ✅ `pages/Login.jsx` - User authentication
- ✅ `pages/Register.jsx` - New user signup
- ✅ `pages/AdminDashboard.jsx` - Admin layout wrapper
- ✅ `pages/AdminHome.jsx` - Admin dashboard with stats
- ✅ `pages/SmartVirtualEnvironment.jsx` - Lectures & Attendance tabs
- ✅ `pages/VideoAnalysisDashboard.jsx` - Video highlights & AI summary
- ✅ `pages/UserHome.jsx` - Student dashboard
- ✅ `pages/SmartAssignment.jsx` - Assignment management
- ✅ `pages/CognivoiceViva.jsx` - Voice learning interface

### Backend Files (Node.js)
- ✅ `server.js` - Express server with MongoDB
- ✅ `package.json` - Dependencies
- ✅ `.env` - Configuration
- ✅ `routes/auth.js` - Authentication routes
- ✅ `routes/users.js` - User CRUD routes
- ✅ `controllers/authController.js` - Auth business logic
- ✅ `middleware/auth.js` - JWT verification middleware
- ✅ `models/User.js` - MongoDB user schema

### Python AI/ML Files
- ✅ `main.py` - FastAPI server with WebSocket
- ✅ `config.py` - Configuration
- ✅ `models.py` - Data models
- ✅ `proctoring_engine.py` - Face detection & analysis
- ✅ `quiz_generator.py` - Quiz creation
- ✅ `requirements.txt` - Python dependencies
- ✅ `lecture_gallery/` - Recorded videos

---

## 🔄 Architecture Overview

```
┌─────────────────────────────────────┐
│  Frontend (React)                   │
│  Port: 5173                         │
│  Framework: Vite + Tailwind         │
│  - Login/Register (Public)          │
│  - Admin Dashboard (Protected)      │
│  - Student Home (Protected)         │
└──────────────┬──────────────────────┘
               │ HTTP API
               ▼
┌─────────────────────────────────────┐
│  Backend (Node.js + Express)        │
│  Port: 5000                         │
│  - JWT Authentication               │
│  - User Management                  │
│  - MongoDB Integration              │
└──────────────┬──────────────────────┘
               │ REST API
               ▼
┌─────────────────────────────────────┐
│  Database (MongoDB)                 │
│  - users collection                 │
│  - Unique email index               │
│  - Hashed passwords (bcryptjs)      │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  AI/ML Services (Python FastAPI)    │
│  Port: 8000                         │
│  - Proctoring Engine               │
│  - Video Analysis                   │
│  - Attendance Tracking              │
└─────────────────────────────────────┘
```

---

## 🚀 Getting Started

### 1. **Frontend Setup**
```bash
cd nglsc-project/frontend
npm install
npm run dev
# Running on http://localhost:5173
```

### 2. **Backend Setup**
```bash
cd nglsc-project/backend
npm install
node server.js
# Running on http://localhost:5000
```

### 3. **Python AI/ML Setup**
```bash
cd nglsc-project/nglsc
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
uvicorn main:app --reload
# Running on http://localhost:8000
```

---

## 🔐 Authentication Flow

### Registration
1. User fills registration form (name, email, password, role)
2. Frontend: `POST /api/auth/register`
3. Backend: Validates, hashes password, creates MongoDB user
4. Backend: Generates JWT token
5. Frontend: Stores token in localStorage
6. Frontend: Redirects to dashboard based on role

### Login
1. User enters credentials (email, password)
2. Frontend: `POST /api/auth/login`
3. Backend: Verifies credentials against MongoDB
4. Backend: Generates JWT token
5. Frontend: Stores token, redirects to dashboard

### Protected Routes
1. Frontend: Checks `localStorage.token`
2. Frontend: `ProtectedRoute` component verifies role
3. Backend: JWT middleware validates token on API calls
4. Automatic redirect to `/login` if unauthorized

---

## 📋 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login & get JWT token
- `GET /api/auth/verify` - Verify token validity

### Users
- `GET /api/users/me` - Get current user
- `GET /api/users` - Get all users (Admin only)
- `PUT /api/users/:id` - Update user
- `DELETE /api/users/:id` - Delete user (Admin only)

### Health
- `GET /api/health` - Server health check

---

## 🗄️ MongoDB Collections

### users Collection
```javascript
{
  _id: ObjectId,
  name: String,
  email: String (unique),
  password: String (hashed),
  role: String ("Student" | "Admin"),
  createdAt: Date,
  updatedAt: Date
}
```

---

## 🔧 Environment Variables

### Backend (.env)
```
PORT=5000
MONGODB_URI=mongodb://localhost:27017/nglsc
JWT_SECRET=your-secret-key-change-in-production
JWT_EXPIRE=7d
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
```

---

## 📚 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Frontend | React | 18.2 |
| Build | Vite | 5.0.8 |
| Styling | Tailwind CSS | 3.3.6 |
| Routing | React Router | 7.10.1 |
| Backend | Node.js | 18+ |
| Server | Express | 4.18.2 |
| Database | MongoDB | 6.3.0 |
| Auth | JWT | 9.1.2 |
| Password | bcryptjs | 2.4.3 |
| AI/ML | FastAPI | - |
| Vision | MediaPipe | - |
| Video | OpenCV | - |

---

## ✅ Project Status

- ✅ Frontend: 9 pages, 14 routes, Tailwind styling
- ✅ Backend: 5 API endpoints, MongoDB ready
- ✅ Authentication: JWT, bcrypt, role-based access
- ✅ Python AI: Proctoring engine, video analysis
- ✅ Documentation: Complete setup guides
- ⏳ Next: MongoDB connection testing, feature integration

---

## 📝 Next Steps

1. **Test MongoDB Connection**
   - Ensure MongoDB is running
   - Update `.env` with correct URI
   - Run backend and test registration

2. **Connect Features**
   - Link quiz system to backend
   - Implement file upload endpoints
   - Add assignment submission API

3. **Deploy to Production**
   - Frontend to Vercel/Netlify
   - Backend to Heroku/Railway
   - MongoDB to Atlas

---

**Project: NGLSC (Next Gen Learning Smart Classroom)**
**Version: 1.0.0**
**Last Updated: January 19, 2026**
