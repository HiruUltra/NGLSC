import mediapipe as mp
try:
    print(f"Mediapipe version: {mp.__version__}")
    print(f"Has solutions: {hasattr(mp, 'solutions')}")
    if hasattr(mp, 'solutions'):
        print(f"Has face_mesh: {hasattr(mp.solutions, 'face_mesh')}")
        if hasattr(mp.solutions, 'face_mesh'):
            print(f"Has FaceMesh class: {hasattr(mp.solutions.face_mesh, 'FaceMesh')}")
except Exception as e:
    print(f"Error checking mediapipe: {e}")
