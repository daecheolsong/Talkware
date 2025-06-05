"""
Utility script to clean up log files
"""

import os
import shutil
from pathlib import Path

def cleanup_logs():
    """Clean up all log files and directories."""
    # Get project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Clean up logs directory
    log_dir = os.path.join(project_root, 'logs')
    if os.path.exists(log_dir):
        print(f"Removing log directory: {log_dir}")
        shutil.rmtree(log_dir, ignore_errors=True)
        
    # Clean up any stray log files in the test directory
    test_dir = os.path.dirname(__file__)
    for item in os.listdir(test_dir):
        if item.endswith('.log') or item.endswith('.gz'):
            file_path = os.path.join(test_dir, item)
            try:
                print(f"Removing log file: {file_path}")
                os.remove(file_path)
            except (OSError, PermissionError) as e:
                print(f"Error removing {file_path}: {e}")

if __name__ == '__main__':
    cleanup_logs()
    print("Log cleanup completed") 