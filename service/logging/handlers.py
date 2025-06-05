"""
Custom logging handlers for Talkware
"""

import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


class CustomRotatingFileHandler(RotatingFileHandler):
    """
    Custom implementation of RotatingFileHandler that includes:
    - Timestamp-based rotation naming
    - Automatic directory creation
    """
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0,
                 encoding=None, delay=False):
        """
        Initialize the handler with custom parameters.
        
        Args:
            filename (str): Log file path
            mode (str): File open mode
            maxBytes (int): Max file size in bytes before rotation
            backupCount (int): Number of backup files to keep
            encoding (str): File encoding
            delay (bool): Delay file creation until first log
        """
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)

    def rotation_filename(self, default_name):
        """
        Generate the rotation filename with timestamp.
        
        Args:
            default_name (str): Default rotation name
            
        Returns:
            str: Rotation filename with timestamp
        """
        dir_name = os.path.dirname(default_name)
        base_name = os.path.basename(self.baseFilename)
        name_parts = os.path.splitext(base_name)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        
        return os.path.join(dir_name, f"{name_parts[0]}.{timestamp}{name_parts[1]}")

    def doRollover(self):
        """
        Perform log rotation with timestamp-based naming.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        # Get new backup filename with timestamp
        dfn = self.rotation_filename(self.baseFilename)
        
        # If backup exists (shouldn't, due to timestamp), remove it
        if os.path.exists(dfn):
            os.remove(dfn)
            
        # Rename current log file to backup filename
        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, dfn)
            
        # Remove old backup files if we have too many
        if self.backupCount > 0:
            dir_name, base_name = os.path.split(self.baseFilename)
            base_name = os.path.splitext(base_name)[0]
            
            # Get list of backup files
            backup_files = [f for f in os.listdir(dir_name)
                          if f.startswith(f"{base_name}.")
                          and f != os.path.basename(self.baseFilename)]
            
            # Sort by timestamp (newest first)
            backup_files.sort(reverse=True)
            
            # Remove old files
            for backup_file in backup_files[self.backupCount:]:
                os.remove(os.path.join(dir_name, backup_file))

        if not self.delay:
            self.stream = self._open()
            
    def _open(self):
        """
        Open the log file, creating the directory if needed.
        
        Returns:
            file: Opened file object
        """
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(self.baseFilename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        return super()._open() 