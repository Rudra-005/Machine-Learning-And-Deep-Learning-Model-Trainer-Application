"""
Session manager for maintaining user sessions and training state
"""
import uuid
from datetime import datetime
from app.utils.logger import logger

class SessionManager:
    """Manage training sessions"""
    
    _sessions = {}
    
    @staticmethod
    def create_session(user_id: str) -> str:
        """
        Create new training session
        
        Args:
            user_id: User identifier
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        SessionManager._sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'status': 'initialized',
            'config': {},
            'results': {},
            'logs': [],
        }
        logger.info(f"Session created: {session_id}")
        return session_id
    
    @staticmethod
    def get_session(session_id: str) -> dict:
        """
        Retrieve session data
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data
        """
        return SessionManager._sessions.get(session_id, {})
    
    @staticmethod
    def update_session(session_id: str, updates: dict) -> None:
        """
        Update session data
        
        Args:
            session_id: Session ID
            updates: Dictionary of updates
        """
        if session_id in SessionManager._sessions:
            SessionManager._sessions[session_id].update(updates)
            logger.debug(f"Session updated: {session_id}")
    
    @staticmethod
    def set_config(session_id: str, config: dict) -> None:
        """
        Set training configuration
        
        Args:
            session_id: Session ID
            config: Training configuration
        """
        if session_id in SessionManager._sessions:
            SessionManager._sessions[session_id]['config'] = config
            logger.info(f"Config set for session: {session_id}")
    
    @staticmethod
    def add_log(session_id: str, message: str) -> None:
        """
        Add log message to session
        
        Args:
            session_id: Session ID
            message: Log message
        """
        if session_id in SessionManager._sessions:
            SessionManager._sessions[session_id]['logs'].append({
                'timestamp': datetime.now().isoformat(),
                'message': message
            })
    
    @staticmethod
    def delete_session(session_id: str) -> None:
        """
        Delete session
        
        Args:
            session_id: Session ID
        """
        if session_id in SessionManager._sessions:
            del SessionManager._sessions[session_id]
            logger.info(f"Session deleted: {session_id}")
