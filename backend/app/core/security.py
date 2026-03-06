"""
Security utilities placeholder.
Extend this module when authentication / authorisation is required.
"""
# Example (JWT with python-jose):
#
# from jose import JWTError, jwt
# from datetime import datetime, timedelta
# import os
#
# SECRET_KEY   = os.getenv("SECRET_KEY", "change-me-in-production")
# ALGORITHM    = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 30
#
# def create_access_token(data: dict) -> str:
#     to_encode = data.copy()
#     expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#
# def verify_token(token: str) -> dict:
#     try:
#         return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#     except JWTError:
#         return {}
