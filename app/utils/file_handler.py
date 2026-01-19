"""
File handling utilities for data upload and storage
"""
import os
import hashlib
from pathlib import Path
import pandas as pd
from app.config import UPLOADS_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS
from app.utils.logger import logger

class FileHandler:
    """Handle file operations for uploads and storage"""
    
    @staticmethod
    def validate_file(file) -> tuple[bool, str]:
        """
        Validate uploaded file
        
        Args:
            file: Uploaded file object (Streamlit UploadedFile)
            
        Returns:
            Tuple of (is_valid, message)
        """
        if file is None:
            return False, "No file provided"
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return False, f"File size exceeds {MAX_FILE_SIZE / (1024*1024):.0f} MB limit"
        
        # Check file extension
        file_ext = Path(file.name).suffix.lstrip('.').lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return False, f"Only {ALLOWED_EXTENSIONS} files are allowed"
        
        return True, "File is valid"
    
    @staticmethod
    def save_file(file, user_id: str = "default") -> tuple[bool, str]:
        """
        Save uploaded file to storage
        
        Args:
            file: Uploaded file object
            user_id: User identifier for organizing files
            
        Returns:
            Tuple of (success, file_path)
        """
        is_valid, message = FileHandler.validate_file(file)
        if not is_valid:
            return False, message
        
        try:
            # Create user directory
            user_dir = UPLOADS_DIR / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            file_hash = hashlib.md5(file.name.encode()).hexdigest()[:8]
            filename = f"{Path(file.name).stem}_{file_hash}.csv"
            filepath = user_dir / filename
            
            # Save file
            with open(filepath, 'wb') as f:
                f.write(file.getbuffer())
            
            logger.info(f"File saved: {filepath}")
            return True, str(filepath)
        
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return False, f"Error saving file: {str(e)}"
    
    @staticmethod
    def load_csv(filepath: str) -> tuple[bool, object]:
        """
        Load CSV file as pandas DataFrame
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Tuple of (success, DataFrame or error message)
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return True, df
        
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return False, f"Error loading CSV: {str(e)}"
