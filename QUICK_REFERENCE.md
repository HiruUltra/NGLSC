# ⚡ NGLSC Quick Reference Card

## 🎯 Project Structure

```
NGLSC/
├── frontend/        🎨 React (Port 5173)
├── backend/         🔧 Node.js (Port 5000)  
└── nglsc/           🤖 Python (Port 8000)
```

---

## 🚀 Quick Start Commands

### 1️⃣ Backend Setup
```bash
cd NGLSC/backend
npm install
node server.js          # ← Runs on port 5000
```

### 2️⃣ Frontend Setup
```bash
cd NGLSC/frontend
npm install
npm run dev             # ← Runs on port 5173
```

### 3️⃣ Python AI Setup
```bash
cd NGLSC/nglsc
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
uvicorn main:app --reload    # ← Runs on port 8000
```

---

## 📝 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/auth/register` | Create new user |
| `POST` | `/api/auth/login` | User login |
| `GET` | `/api/auth/verify` | Verify JWT token |
| `GET` | `/api/users/me` | Get current user |
| `GET` | `/api/users` | Get all users (Admin) |
| `PUT` | `/api/users/:id` | Update user |
| `DELETE` | `/api/users/:id` | Delete user (Admin) |
| `GET` | `/api/health` | Server health check |

---

## 🔐 Test Credentials

### Register as Admin
- Email: `admin@nglsc.com`
- Password: `password123`
- Role: **Admin**

### Register as Student
- Email: `student@nglsc.com`
- Password: `password123`
- Role: **Student**

---

## 🗺️ Frontend Routes

### Public
- `/login` - Login page
- `/register` - Registration

### Admin (Protected)
- `/admin/home` - Dashboard
- `/admin/smart-virtual` - Lectures
- `/admin/video-analysis` - Video analysis
- `/admin/marks` - Marks management
- `/admin/users` - User management

### Student (Protected)
- `/home` - Dashboard
- `/quiz` - Quiz system
- `/assignments` - Assignments
- `/cognivoice` - Voice learning

---

## 🗄️ MongoDB Setup

### Local MongoDB
```bash
# Connection string
mongodb://localhost:27017/nglsc
```

### MongoDB Atlas (Cloud)
```bash
# Connection string format
mongodb+srv://username:password@cluster.mongodb.net/nglsc
```

**Update in:** `backend/.env`

---

## 📄 Environment Variables

**File:** `backend/.env`

```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/nglsc
JWT_SECRET=your-secret-key
JWT_EXPIRE=7d
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
```

---

## 🔧 Backend File Structure

```
backend/
├── server.js                 ← Start here
├── package.json
├── .env
├── routes/
│   ├── auth.js
│   └── users.js
├── controllers/
│   └── authController.js
├── middleware/
│   └── auth.js
└── models/
    └── User.js
```

---

## 📂 Frontend File Structure

```
frontend/src/
├── App.jsx                   ← Routing (14 routes)
├── pages/
│   ├── Login.jsx
│   ├── Register.jsx
│   ├── AdminDashboard.jsx
│   ├── AdminHome.jsx
│   ├── UserHome.jsx
│   ├── SmartVirtualEnvironment.jsx
│   ├── VideoAnalysisDashboard.jsx
│   ├── SmartAssignment.jsx
│   └── CognivoiceViva.jsx
├── components/               ← Reusable components
├── context/                  ← Theme provider
└── hooks/                    ← Custom hooks
```

---

## 🐍 Python Files

```
nglsc/
├── main.py                  ← FastAPI server
├── config.py                ← Settings
├── models.py                ← Data models
├── proctoring_engine.py     ← Face detection
├── quiz_generator.py        ← Quiz logic
└── requirements.txt
```

---

## 🎨 Authentication Flow

```
User Registration
    ↓
POST /api/auth/register
    ↓
Hash password (bcryptjs)
    ↓
Create user in MongoDB
    ↓
Generate JWT token
    ↓
Return token + user data
    ↓
Store in localStorage
    ↓
Redirect to dashboard
```

---

## 🔑 JWT Token Format

```javascript
{
  userId: "507f1f77bcf86cd799439011",
  email: "user@example.com",
  role: "Student" | "Admin",
  iat: 1705684800,
  exp: 1706289600
}
```

---

## 📊 Database Schema

### Users Collection
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

## 🛠️ Troubleshooting

### MongoDB won't connect
✅ Check if MongoDB is running
✅ Verify connection string in `.env`
✅ Check if database exists

### Port already in use
✅ Change PORT in `.env`
✅ Kill process using the port

### npm install fails
✅ Clear cache: `npm cache clean --force`
✅ Delete node_modules: `rm -r node_modules`
✅ Reinstall: `npm install`

### Frontend can't reach backend
✅ Ensure backend is running on port 5000
✅ Check CORS in server.js
✅ Verify FRONTEND_URL in `.env`

---

## 📦 Key Dependencies

| Package | Version | Use |
|---------|---------|-----|
| express | ^4.18.2 | Backend server |
| mongodb | ^6.3.0 | Database |
| jsonwebtoken | ^9.1.2 | JWT auth |
| bcryptjs | ^2.4.3 | Password hash |
| cors | ^2.8.5 | Cross-origin |
| react | 18.2 | Frontend |
| vite | 5.0.8 | Build tool |
| tailwindcss | 3.3.6 | Styling |
| react-router | 7.10.1 | Routing |

---

## 🎯 Development Checklist

- [ ] Install backend dependencies
- [ ] Install frontend dependencies
- [ ] Set up MongoDB (local or Atlas)
- [ ] Start backend server
- [ ] Start frontend dev server
- [ ] Test registration endpoint
- [ ] Test login endpoint
- [ ] Verify user in MongoDB
- [ ] Test protected routes
- [ ] Test admin dashboard
- [ ] Test student dashboard

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `PROJECT_STRUCTURE.md` | Complete project overview |
| `SETUP_GUIDE.md` | Installation & setup steps |
| `DIRECTORY_TREE.md` | Visual folder structure |
| `ARCHITECTURE.md` | System architecture diagrams |
| `QUICK_REFERENCE.md` | This file! |

---

## 🌐 URLs After Startup

| Service | URL |
|---------|-----|
| Frontend | `http://localhost:5173` |
| Backend | `http://localhost:5000` |
| Health Check | `http://localhost:5000/api/health` |
| Python AI | `http://localhost:8000` |
| MongoDB | `localhost:27017` |

---

## 💡 Pro Tips

1. **Use Postman** to test API endpoints
2. **Check browser console** for frontend errors
3. **Check terminal logs** for backend errors
4. **MongoDB Compass** to visualize database
5. **VS Code REST Client** for API testing

---

## 🚀 Deployment Checklist

- [ ] Change JWT_SECRET for production
- [ ] Set NODE_ENV=production
- [ ] Use MongoDB Atlas (not local)
- [ ] Configure environment variables
- [ ] Test all routes
- [ ] Build frontend: `npm run build`
- [ ] Deploy to Vercel/Netlify
- [ ] Deploy backend to Heroku/Railway
- [ ] Set up CI/CD pipeline
- [ ] Monitor logs

---

**Keep this card handy for quick reference!** ⭐
