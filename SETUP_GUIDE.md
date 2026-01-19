# 🚀 NGLSC Project Setup Guide

## Project Location
```
c:\Users\dilha\OneDrive\Documents\Desktop\Ngs-croom\NGLSC
```

## ✨ Complete Folder Structure

Your project is now organized with **three main components**:

### 1️⃣ **Frontend** (React + Vite + Tailwind)
```
frontend/
├── src/
│   ├── pages/                (9 authentication & dashboard pages)
│   ├── components/           (reusable React components)
│   ├── context/              (theme context)
│   ├── hooks/                (custom hooks)
│   ├── App.jsx              (14 routes with role-based protection)
│   └── main.jsx
├── package.json
├── vite.config.js
├── tailwind.config.js
└── index.html
```

**Location:** `NGLSC/frontend/`

---

### 2️⃣ **Backend** (Node.js + Express + MongoDB)
```
backend/
├── server.js                 (Express server, MongoDB connection)
├── package.json             (dependencies installed)
├── .env                     (MongoDB URI, JWT secrets)
│
├── routes/
│   ├── auth.js             (register, login, verify)
│   └── users.js            (CRUD operations)
│
├── controllers/
│   └── authController.js   (business logic)
│
├── middleware/
│   └── auth.js             (JWT verification)
│
├── models/
│   └── User.js             (MongoDB schema)
│
└── venv/                    (Python venv - legacy, can delete)
```

**Location:** `NGLSC/backend/`

---

### 3️⃣ **Python AI/ML** (FastAPI + MediaPipe + OpenCV)
```
nglsc/
├── main.py                 (FastAPI server, WebSocket)
├── config.py               (settings)
├── models.py               (data models)
├── proctoring_engine.py    (face detection, head pose)
├── quiz_generator.py       (quiz logic)
├── requirements.txt        (Python packages)
│
├── models/                 (pre-trained ML models)
└── lecture_gallery/        (recorded videos)
```

**Location:** `NGLSC/nglsc/`

---

## 🎯 What's Ready to Use

### ✅ Frontend
- 9 complete pages (Login, Register, Admin Dashboard, Student Home, etc.)
- 14 routes with protected access control
- Professional Tailwind CSS styling
- Dark/Light theme support
- Role-based navigation (Admin vs Student)

### ✅ Backend
- Express server with CORS enabled
- MongoDB integration ready
- 5 API endpoints (register, login, verify, get users, CRUD)
- JWT authentication with bcryptjs password hashing
- User role management (Student/Admin)
- Protected routes with middleware

### ✅ Python AI/ML
- FastAPI with WebSocket support
- Proctoring engine (face detection, attention tracking)
- Video analysis capabilities
- Quiz generation system
- Attendance tracking

---

## 🔧 Installation & Running

### **Step 1: Install Node.js Dependencies (Backend)**
```bash
cd NGLSC/backend
npm install
```

**Expected output:** ✅ installed 45 packages

---

### **Step 2: Install Frontend Dependencies**
```bash
cd NGLSC/frontend
npm install
```

**Expected output:** ✅ installed 50+ packages

---

### **Step 3: Set Up Python Environment**
```bash
cd NGLSC/nglsc
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### **Terminal 1: Start Backend**
```bash
cd NGLSC/backend
npm install
node server.js
```
✅ Server will run on `http://localhost:5000`

---

### **Terminal 2: Start Frontend**
```bash
cd NGLSC/frontend
npm run dev
```
✅ App will run on `http://localhost:5173`

---

### **Terminal 3: Start Python AI/ML**
```bash
cd NGLSC/nglsc
.\venv\Scripts\Activate
uvicorn main:app --reload
```
✅ FastAPI will run on `http://localhost:8000`

---

## 📝 Test the Application

### 1. **Open Frontend**
```
http://localhost:5173
```

### 2. **Register as Admin**
- Email: `admin@nglsc.com`
- Password: `password123`
- Role: **Admin**
- Click "Sign Up"

### 3. **Login & Explore**
- Should redirect to `/admin/home`
- See admin dashboard with sidebar
- Try different pages (Virtual Environment, Video Analysis, etc.)

### 4. **Register as Student**
- Email: `student@nglsc.com`
- Password: `password123`
- Role: **Student**
- Should redirect to `/home` with student features

---

## 🗄️ MongoDB Setup (Important!)

The backend expects MongoDB running. You have two options:

### Option A: **Local MongoDB** (Recommended for Development)
1. [Download & Install MongoDB](https://www.mongodb.com/try/download/community)
2. Update `.env` file:
```
MONGODB_URI=mongodb://localhost:27017/nglsc
```
3. Start MongoDB service
4. Run backend

### Option B: **MongoDB Atlas** (Cloud - Recommended for Production)
1. [Create free account](https://www.mongodb.com/cloud/atlas)
2. Create a cluster
3. Get connection string
4. Update `.env`:
```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/nglsc?retryWrites=true&w=majority
```
5. Run backend

---

## 🔐 Environment Variables

**File:** `NGLSC/backend/.env`

```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/nglsc
JWT_SECRET=your-secret-key-change-in-production
JWT_EXPIRE=7d
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
```

⚠️ **Security Note:** Change `JWT_SECRET` in production!

---

## 📊 API Endpoints (Backend)

### Health Check
```
GET http://localhost:5000/api/health
```

### Authentication
```
POST http://localhost:5000/api/auth/register
POST http://localhost:5000/api/auth/login
GET http://localhost:5000/api/auth/verify
```

### Users
```
GET http://localhost:5000/api/users/me
GET http://localhost:5000/api/users              (Admin only)
PUT http://localhost:5000/api/users/:id
DELETE http://localhost:5000/api/users/:id       (Admin only)
```

---

## 🎨 Frontend Routes

### Public Routes
- `/login` - Login page
- `/register` - Registration page

### Admin Routes (Protected)
- `/admin/home` - Dashboard with stats
- `/admin/smart-virtual` - Lectures & Attendance
- `/admin/video-analysis` - Video highlights
- `/admin/marks` - Student marks
- `/admin/users` - User management

### Student Routes (Protected)
- `/home` - Dashboard with features
- `/quiz` - Quiz system
- `/assignments` - Assignment management
- `/cognivoice` - Voice learning

---

## ✅ Verification Checklist

- [ ] **Frontend** installed at `NGLSC/frontend/`
- [ ] **Backend** installed at `NGLSC/backend/`
- [ ] **Python** files organized in `NGLSC/nglsc/`
- [ ] **MongoDB** ready (local or Atlas)
- [ ] **Backend** starts on port 5000
- [ ] **Frontend** starts on port 5173
- [ ] **Registration** works (user created in MongoDB)
- [ ] **Login** works (JWT token generated)
- [ ] **Protected routes** redirect unauthorized users
- [ ] **Admin dashboard** shows all features

---

## 🐛 Troubleshooting

### Issue: MongoDB connection error
**Solution:** Ensure MongoDB is running
```bash
# Check if MongoDB is running (Windows)
net start MongoDB
# Or start MongoDB service
```

### Issue: Port 5000 already in use
**Solution:** Change PORT in `.env` or kill process using port
```bash
# PowerShell - kill process on port 5000
Get-Process | Where-Object {$_.ProcessName -like "*node*"} | Stop-Process
```

### Issue: npm install fails
**Solution:** Clear npm cache
```bash
npm cache clean --force
npm install
```

### Issue: Frontend doesn't connect to backend
**Solution:** Check CORS in backend
- Backend has CORS enabled for `http://localhost:5173`
- Check `.env` FRONTEND_URL setting

---

## 📚 Documentation Files

- **PROJECT_STRUCTURE.md** - Complete project overview
- **ARCHITECTURE.md** - Visual architecture diagrams
- **QUICKSTART.md** - 5-minute setup guide
- **README.md** - Full documentation

---

## 🎯 Next Steps

1. ✅ **Install Dependencies**
   ```bash
   cd backend && npm install
   cd ../frontend && npm install
   ```

2. ✅ **Set Up MongoDB**
   - Install locally or use Atlas

3. ✅ **Run All Servers**
   - Backend (port 5000)
   - Frontend (port 5173)
   - Python AI (port 8000)

4. ✅ **Test Authentication**
   - Register new user
   - Login
   - Verify MongoDB user creation

5. ✅ **Explore Features**
   - Admin dashboard
   - Student home page
   - Video analysis
   - Quiz system

---

**Status:** 🎉 **PROJECT STRUCTURE COMPLETE & READY TO USE!**

**Setup Time:** ~10 minutes
**Start Time:** ~2 minutes (after npm install)

---

*For detailed API documentation, see QUICKSTART.md*
*For architecture diagrams, see ARCHITECTURE.md*
