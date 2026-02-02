# db_factory.py
"""
Database Factory - Platform-independent database connections
Supports: PostgreSQL, SQLite
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class DatabaseFactory:
    """Factory for creating database connections"""
    
    @staticmethod
    def get_connection(db_type: Optional[str] = None):
        """
        Get database connection based on DB_TYPE
        
        Args:
            db_type: Database type (postgresql, sqlite)
            
        Returns:
            Database connection object
        """
        db_type = db_type or os.getenv("DB_TYPE", "sqlite").lower()
        
        if db_type == "postgresql":
            return DatabaseFactory._get_postgresql_connection()
        elif db_type == "sqlite":
            return DatabaseFactory._get_sqlite_connection()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    @staticmethod
    def _get_postgresql_connection():
        """Get PostgreSQL connection"""
        import psycopg2
        
        config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "dbname": os.getenv("DB_NAME", "load_forecasting"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "")
        }
        
        return psycopg2.connect(**config)
    
    @staticmethod
    def _get_sqlite_connection():
        """Get SQLite connection"""
        import sqlite3
        
        db_path = os.getenv("SQLITE_DB_PATH", "./data/load_forecasting.db")
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(exist_ok=True)
        
        return sqlite3.connect(db_path)
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get database configuration"""
        db_type = os.getenv("DB_TYPE", "sqlite").lower()
        
        if db_type == "postgresql":
            return {
                "type": "postgresql",
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "dbname": os.getenv("DB_NAME", "load_forecasting"),
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", "")
            }
        else:  # sqlite
            return {
                "type": "sqlite",
                "path": os.getenv("SQLITE_DB_PATH", "./data/load_forecasting.db")
            }

    @staticmethod
    def get_engine(db_type: Optional[str] = None):
        """Return a lightweight SQLAlchemy engine string when available or None.

        This helper does NOT require SQLAlchemy at import-time; it will only
        attempt to create an engine if SQLAlchemy is installed. It is useful
        for passing a supported "connectable" to pandas.to_sql when available.
        """
        db_type = db_type or os.getenv("DB_TYPE", "sqlite").lower()
        cfg = DatabaseFactory.get_config()

        try:
            from sqlalchemy import create_engine
        except Exception:
            return None

        if cfg["type"] == "postgresql":
            user = cfg.get("user") or "postgres"
            password = cfg.get("password", "")
            host = cfg.get("host", "localhost")
            port = cfg.get("port", 5432)
            dbname = cfg.get("dbname")
            url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
            return create_engine(url)
        else:
            path = cfg.get("path", "./data/load_forecasting.db")
            return create_engine(f"sqlite:///{path}")


def get_db_connection():
    """Convenience function to get database connection"""
    return DatabaseFactory.get_connection()