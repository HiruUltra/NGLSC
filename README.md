# Next-Generation Smart Classroom

This project implements an AI-powered smart classroom system...
# Next-Generation Smart Classroom System

## 📘 Project Overview

The **Next-Generation Smart Classroom** is an intelligent, AI-driven educational platform designed to enhance examination integrity, lecture efficiency, and assignment management in modern classrooms. The system integrates computer vision, voice analysis, and smart automation to create a more secure, efficient, and user-centered learning environment.
The Next-Generation Smart Classroom is an intelligent, AI-driven educational platform designed to enhance examination integrity, lecture efficiency, and assignment management in modern classrooms. The system integrates computer vision, voice analysis, and smart automation to create a more secure, efficient, and user-centered learning environment.
This project addresses major limitations in traditional classroom systems by introducing four key smart components developed by different team members.

## 🔹 Key Features
### 1. Camera-Based Online Examination Monitoring
- Students answer exam questions within a given time while facing the camera.  
- The system detects abnormal behavior such as talking with others and issues live warnings.  
- Ensures fairness and academic integrity.  

### 2. Smart Lecture Recording Optimization
- Recorded lectures are automatically processed to remove silent and inactive segments.  
- Teacher movements and important moments are highlighted.  
- Provides shorter, more effective lecture recordings for students.  

### 3. Voice-Based Examination System
- An intelligent agent asks questions to students.  
- Students answer verbally.  
- The system analyzes voice tempo and patterns to identify confidence or nervousness.  
- Supports alternative and interactive assessment methods.  

### 4. Smart Assignment Management with Diagram Digitization
- Students complete assignments with 10 MCQs and two diagram questions before deadlines.  
- Handwritten diagrams can be photographed and uploaded.  
- The system converts handwritten diagrams into digital format.  
- Encourages productive use of classroom time and improves submission efficiency.  

## 🎯 Project Goal

The goal of this research is to develop a **smart, secure, and user-centered classroom environment** that improves learning quality, assessment fairness, and academic productivity using modern intelligent technologies.

## 🌟 Impact

The Next-Generation Smart Classroom benefits:

- **Students** by improving learning efficiency and assessment flexibility  
- **Lecturers** by optimizing lecture delivery and evaluation methods  
- **Institutions** by ensuring academic integrity and modern digital education standards

## 🔹 Component 2: Conversational Intelligence & Voice Technologies 
**Contributor:** IT22229816 – Silva L.J.U 

---
### 📘  Overview

CogniVoice is a Conversational Intelligence Module developed as part of the Next-Gen Smart Classroom Research Project.
It functions as an intelligent AI-based Viva Assistant, designed to modernize student assessment in A/L tuition and smart classroom environments.

Integrated into a Smart Podium, CogniVoice enables hands-free, unbiased, and real-time oral assessments by combining speech recognition, NLP, deep learning, and machine learning techniques.

🔐 Students are identified using facial recognition, after which their personal performance dashboard is automatically loaded.
🎯 The system then generates 10 subject-specific questions, evaluates spoken answers, and provides instant grading and feedback.

---

### ✨ Key Features

🧠 Automate short-answer and viva assessments
🎤 Evaluate both answer correctness and speaking confidence
📊 Reduce examiner bias and manual grading effort
🏫 Enable scalable assessments in smart classrooms
✅ Automated Text Grading
 - Fuzzy keyword matching with 82% similarity threshold
✅ Voice Answer Grading
 - Speech-to-text conversion + automatic evaluation
✅ Voice Confidence Detection
 - Analyzes speech patterns to classify confidence levels
✅ Real-Time Transcription
 - Powered by OpenAI Whisper (Base Model – 74M parameters)
✅ Multi-Format Audio Support
 - AV, MP3, M4A, OGG, WebM (auto-conversion supported)
✅ Silence Detection
 - Filters empty or no-speech audio inputs

 ---

 ### ⚙️ Dependencies

Flask==3.0.0                 # REST API backend
torch==2.0.0                 # Required for Whisper
openai-whisper==20231117     # Speech-to-text
tensorflow==2.15.0           # Required for YAMNet
tensorflow-hub==0.15.0       # Load pre-trained YAMNet
scikit-learn==1.3.2          # ML models & pipelines

numpy==1.24.3                # Numerical operations
pandas==2.1.3                # Question bank handling
joblib==1.3.2                # Model serialization

librosa==0.10.1              # Audio processing
soundfile==0.12.1            # Audio I/O

---

### 📚 Libraries Used

- whisper	 
- tensorflow 
- librosa	 
- pandas	
- numpy	
- joblib	
- sklearn.feature_extraction	
- sklearn.model_selection	
- matplotlib	

----

## 🔹 Component 3: Visual Intelligent and Extraction Management  
**Contributor:** IT222111312 – De Silva L.S  

---

### 📘 Overview

This module enables students to complete assignments within classroom extra time before deadlines using a smart digital platform. Each assignment contains **10 MCQ questions and two diagram questions**. Students can draw diagrams by hand, capture a photo, upload it, and convert the handwritten diagram into a digital format using the system.

This ensures timely submission, better organization, and improved digital learning support. The system encourages productivity, reduces last-minute submissions, and supports both handwritten and digital learning styles.

---

### ✨ Key Features

- Deadline-based assignment submission system  
- MCQ-based assessment with automatic validation  
- Handwritten diagram upload using camera  
- Handwritten diagram to digital diagram conversion  
- Extra classroom time utilization tracking  
- Secure student submission management  
- Digital storage of assignments for future reference  

---

### ⚙️ Dependencies

- Python 3.9+  
- OpenCV (for image processing)  
- TensorFlow / PyTorch (for diagram recognition model)  
- Flask / FastAPI (for backend API)  
- MySQL / MongoDB (for database)  
- Node.js (if frontend uses React / Angular)  

---

### 📚 Libraries Used

- `flask` / `fastapi` – REST API development  
- `sqlalchemy` / `pymongo` – Database connectivity  
- `jwt` – Authentication and authorization  

---


![Next Gen Classroom System – System Overview](Architectural-diagram.jpeg)


