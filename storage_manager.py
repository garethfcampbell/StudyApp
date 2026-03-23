import os
import json
import uuid
import logging
from datetime import datetime, timedelta

class StorageManager:
    def __init__(self, storage_dir='session_storage'):
        """Initialize storage manager with specified directory"""
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
    def generate_session_id(self):
        """Generate a unique session ID"""
        return str(uuid.uuid4())
    
    def get_session_file_path(self, session_id, content_type):
        """Get file path for session content"""
        return os.path.join(self.storage_dir, f"{session_id}_{content_type}.json")
    
    def store_content(self, session_id, content_type, content):
        """Store content for a session"""
        try:
            file_path = self.get_session_file_path(session_id, content_type)
            
            data = {
                'session_id': session_id,
                'content_type': content_type,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logging.debug(f"Stored content for session {session_id}, type {content_type}")
            
        except Exception as e:
            logging.error(f"Error storing content: {e}")
            raise Exception(f"Failed to store content: {str(e)}")
    
    def retrieve_content(self, session_id, content_type):
        """Retrieve content for a session"""
        try:
            file_path = self.get_session_file_path(session_id, content_type)
            
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if content has expired
            expires_at = datetime.fromisoformat(data.get('expires_at', datetime.now().isoformat()))
            if datetime.now() > expires_at:
                self.delete_content(session_id, content_type)
                return None
            
            logging.debug(f"Retrieved content for session {session_id}, type {content_type}")
            return data.get('content')
            
        except Exception as e:
            logging.error(f"Error retrieving content: {e}")
            return None
    
    def delete_content(self, session_id, content_type):
        """Delete content for a session"""
        try:
            file_path = self.get_session_file_path(session_id, content_type)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.debug(f"Deleted content for session {session_id}, type {content_type}")
                
        except Exception as e:
            logging.error(f"Error deleting content: {e}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired session files"""
        try:
            current_time = datetime.now()
            deleted_count = 0
            
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.storage_dir, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        expires_at = datetime.fromisoformat(data.get('expires_at', current_time.isoformat()))
                        
                        if current_time > expires_at:
                            os.remove(file_path)
                            deleted_count += 1
                            
                    except Exception as e:
                        logging.error(f"Error processing file {filename}: {e}")
                        # Remove corrupted files
                        os.remove(file_path)
                        deleted_count += 1
            
            if deleted_count > 0:
                logging.info(f"Cleaned up {deleted_count} expired session files")
                
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    def get_session_info(self, session_id):
        """Get information about a session"""
        try:
            session_files = []
            
            for filename in os.listdir(self.storage_dir):
                if filename.startswith(session_id) and filename.endswith('.json'):
                    file_path = os.path.join(self.storage_dir, filename)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    session_files.append({
                        'content_type': data.get('content_type'),
                        'timestamp': data.get('timestamp'),
                        'expires_at': data.get('expires_at')
                    })
            
            return {
                'session_id': session_id,
                'files': session_files,
                'total_files': len(session_files)
            }
            
        except Exception as e:
            logging.error(f"Error getting session info: {e}")
            return None
