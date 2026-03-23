from database import postgres_db as db
from datetime import datetime, timedelta
import json

class SessionData(db.Model):
    """PostgreSQL model for storing session data - replaces ReplitDB session storage"""
    __tablename__ = 'session_data'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(128), nullable=False, index=True)
    content_type = db.Column(db.String(64), nullable=False)
    content = db.Column(db.Text, nullable=False)  # JSON stored as text
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    
    # Composite index for efficient lookups
    __table_args__ = (
        db.Index('idx_session_content', 'session_id', 'content_type'),
        db.Index('idx_expires_at', 'expires_at'),
    )
    
    def __init__(self, session_id, content_type, content, expires_hours=24):
        self.session_id = session_id
        self.content_type = content_type
        self.content = json.dumps(content, ensure_ascii=False)
        self.timestamp = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
    
    def get_content(self):
        """Return parsed JSON content"""
        try:
            return json.loads(self.content)
        except (json.JSONDecodeError, TypeError):
            return None
    
    def set_content(self, content):
        """Set content as JSON"""
        self.content = json.dumps(content, ensure_ascii=False)
        self.timestamp = datetime.utcnow()
    
    def is_expired(self):
        """Check if this session data has expired"""
        return datetime.utcnow() > self.expires_at

class TaskStatus(db.Model):
    """PostgreSQL model for tracking asynchronous task status - replaces ReplitDB task storage"""
    __tablename__ = 'task_status'
    
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(128), unique=True, nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default='pending')  # pending, complete, failed
    success = db.Column(db.Boolean, nullable=True)
    data = db.Column(db.Text, nullable=True)  # JSON stored as text
    error = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    
    # Index for cleanup
    __table_args__ = (
        db.Index('idx_task_status', 'status'),
        db.Index('idx_task_expires', 'expires_at'),
    )
    
    def __init__(self, task_id, status='pending', expires_hours=1):
        self.task_id = task_id
        self.status = status
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
    
    def get_data(self):
        """Return parsed JSON data"""
        if not self.data:
            return None
        try:
            return json.loads(self.data)
        except (json.JSONDecodeError, TypeError):
            return None
    
    def set_data(self, data):
        """Set data as JSON"""
        if data is not None:
            self.data = json.dumps(data, ensure_ascii=False)
        else:
            self.data = None
        self.updated_at = datetime.utcnow()
    
    def set_complete(self, success=True, data=None, error=None):
        """Mark task as complete with result"""
        self.status = 'complete'
        self.success = success
        if data is not None:
            self.set_data(data)
        if error:
            self.error = error
        self.updated_at = datetime.utcnow()
    
    def set_failed(self, error):
        """Mark task as failed with error"""
        self.status = 'failed'
        self.success = False
        self.error = error
        self.updated_at = datetime.utcnow()
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization (replaces ReplitDB dict conversion)"""
        result = {
            'status': self.status,
        }
        
        if self.success is not None:
            result['success'] = self.success
        
        if self.data:
            result['data'] = self.get_data()
        
        if self.error:
            result['error'] = self.error
            
        return result
    
    def is_expired(self):
        """Check if this task has expired"""
        return datetime.utcnow() > self.expires_at