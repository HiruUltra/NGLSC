"""
Database configuration placeholder.
Extend this module when persistent storage is added (e.g. PostgreSQL via SQLAlchemy).
"""
# Example (SQLAlchemy + asyncpg):
#
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
# from sqlalchemy.orm import sessionmaker, DeclarativeBase
# import os
#
# DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./nglsc.db")
# engine = create_async_engine(DATABASE_URL, echo=False)
# AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
#
# class Base(DeclarativeBase):
#     pass
#
# async def get_db():
#     async with AsyncSessionLocal() as session:
#         yield session
