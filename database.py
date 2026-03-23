"""
Database configuration and setup for PostgreSQL
Handles SQLAlchemy initialization and avoids circular imports
"""
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy
postgres_db = SQLAlchemy(model_class=Base)