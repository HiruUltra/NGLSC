from motor.motor_asyncio import AsyncIOMotorClient
import os

# MongoDB connection settings
MONGODB_URL = "mongodb://localhost:27017" # Default local connection
DATABASE_NAME = "proctoring_db"

client = None
db = None

async def connect_to_mongo():
    global client, db
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    print(f"Connected to MongoDB: {DATABASE_NAME}")

async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("MongoDB connection closed")

def get_database():
    return db
