"""
Test log rotation functionality based on maximum size configuration
"""

import os
import sys
import re
import pytest
import logging
import shutil
from pathlib import Path
from datetime import datetime
import time

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from service.logging.config import LogConfig
from service.logging.handlers import CustomRotatingFileHandler
from service.logging.formatters import CustomJSONFormatter

class TestLogRotation:
    def setUp(self):
        """Clean up logs directory before each test"""
        log_dir = os.path.join(project_root, 'logs')
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Remove all files in logs directory
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        """Fixture to run setUp before each test"""
        self.setUp()
        yield  # This is where the testing happens

    @pytest.fixture
    def config(self):
        """Load test configuration"""
        config_path = os.path.join(project_root, 'config', 'test_application.yml')
        log_config = LogConfig(config_path)
        log_config.load_config()
        return log_config

    @pytest.fixture
    def log_dir(self, config):
        """Set up the log directory"""
        log_dir = config.config['logging']['directory']['base']
        
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        return log_dir

    @pytest.fixture
    def json_formatter(self):
        """Create JSON formatter for logs"""
        return CustomJSONFormatter()

    def _parse_size(self, size_str):
        """Parse size string (e.g., '100M', '200M') to bytes"""
        units = {'K': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
        size = int(size_str[:-1])
        unit = size_str[-1]
        return size * units[unit]

    def generate_data(self, target_size):
        """Generate test data smaller than target size"""
        # Calculate safe message size (about 80% of target to ensure rotation)
        safe_size = int(target_size * 0.8)
        base_msg = "Test log message with padding: "
        
        # Ensure we don't create a message larger than safe_size
        available_size = safe_size - len(base_msg) - 10  # Buffer for JSON formatting
        if available_size < 0:
            available_size = 0
            
        padding = "X" * available_size
        return base_msg + padding

    def verify_log_rotation(self, log_dir, base_filename, max_bytes, backup_count):
        """Verify log rotation results"""
        # Verify current log file exists and is not empty
        current_log = os.path.join(log_dir, base_filename)
        assert os.path.exists(current_log), f"Current log file {base_filename} not found"
        assert os.path.getsize(current_log) > 0, f"Current log file {base_filename} is empty"

        # Get all rotated files
        name_without_ext = os.path.splitext(base_filename)[0]
        backup_files = [f for f in os.listdir(log_dir)
                       if f.startswith(name_without_ext + ".")
                       and f != base_filename
                       and f.endswith('.log')]

        # Verify we have exactly backup_count backup files
        assert len(backup_files) == backup_count, f"Expected {backup_count} backup files for {base_filename}, but found {len(backup_files)}"

        # Verify timestamp format in backup files
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}-\d{6}'
        for backup_file in backup_files:
            # Extract timestamp from filename
            match = re.search(timestamp_pattern, backup_file)
            assert match, f"Invalid backup filename format: {backup_file}"

            # Verify file size
            backup_path = os.path.join(log_dir, backup_file)
            file_size = os.path.getsize(backup_path)
            assert file_size <= max_bytes, f"Backup file {backup_file} exceeds max size: {file_size} > {max_bytes}"

            # Verify file content is valid JSON
            with open(backup_path, 'r') as f:
                for line in f:
                    try:
                        import json
                        json.loads(line.strip())
                    except json.JSONDecodeError:
                        assert False, f"Invalid JSON format in file {backup_file}"

    def write_test_logs(self, logger, max_bytes, backup_count):
        """Write enough logs to cause multiple rotations"""
        # Generate messages that will be about 80% of max_bytes
        data = self.generate_data(int(max_bytes * 0.8))
        
        # Write enough messages to cause the specified number of rotations
        # Each rotation requires about 2 messages, so multiply by 3 for safety
        messages_to_write = backup_count * 3
        max_attempts = messages_to_write * 2  # Prevent infinite loops
        
        attempts = 0
        messages_written = 0
        
        while messages_written < messages_to_write and attempts < max_attempts:
            logger.info(f"{data} - Message {messages_written}")
            
            # Add a longer delay to ensure rotation completes
            time.sleep(0.2)  # Increased base delay
            
            # Every few messages, add an extra delay and force a handler flush
            if messages_written % 3 == 0:
                time.sleep(0.3)  # Additional delay every 3 messages
                for handler in logger.handlers:
                    handler.flush()
            
            messages_written += 1
            attempts += 1
            
        if attempts >= max_attempts:
            logger.warning(f"Max attempts ({max_attempts}) reached while writing logs")
            
        # Final delay to ensure last rotations complete
        time.sleep(0.5)

    def setup_handler(self, logger, handler_config, test_config, log_dir, json_formatter):
        """Set up a log handler with the correct configuration"""
        # Use the filename from test_config but place it in the test log_dir
        filename = os.path.basename(handler_config['filename'])
        log_file = os.path.join(log_dir, filename)
        
        max_bytes = handler_config['maxBytes']
        backup_count = handler_config['backupCount']
        
        # Remove any existing handlers
        logger.handlers = []
        
        handler = CustomRotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        handler.setFormatter(json_formatter)
        logger.addHandler(handler)
        
        return max_bytes, backup_count

    def test_app_log_rotation(self, config, log_dir, json_formatter):
        """Test app.log rotation"""
        logger = logging.getLogger('test_app_rotation')
        logger.setLevel(logging.INFO)
        
        handler_config = config.config['logging']['handlers']['app_handler']
        test_config = config.config['logging']['files']['app']
        
        max_bytes, backup_count = self.setup_handler(
            logger, handler_config, test_config, log_dir, json_formatter
        )
        
        self.write_test_logs(logger, max_bytes, backup_count)
        self.verify_log_rotation(log_dir, test_config['filename'], 
                               max_bytes, backup_count)

    def test_inference_log_rotation(self, config, log_dir, json_formatter):
        """Test inference.log rotation"""
        logger = logging.getLogger('test_inference_rotation')
        logger.setLevel(logging.INFO)
        
        handler_config = config.config['logging']['handlers']['inference_handler']
        test_config = config.config['logging']['files']['inference']
        
        max_bytes, backup_count = self.setup_handler(
            logger, handler_config, test_config, log_dir, json_formatter
        )
        
        self.write_test_logs(logger, max_bytes, backup_count)
        self.verify_log_rotation(log_dir, test_config['filename'], 
                               max_bytes, backup_count)

    def test_access_log_rotation(self, config, log_dir, json_formatter):
        """Test access.log rotation"""
        logger = logging.getLogger('test_access_rotation')
        logger.setLevel(logging.INFO)
        
        handler_config = config.config['logging']['handlers']['access_handler']
        test_config = config.config['logging']['files']['access']
        
        max_bytes, backup_count = self.setup_handler(
            logger, handler_config, test_config, log_dir, json_formatter
        )
        
        self.write_test_logs(logger, max_bytes, backup_count)
        self.verify_log_rotation(log_dir, test_config['filename'], 
                               max_bytes, backup_count)

    def test_error_log_rotation(self, config, log_dir, json_formatter):
        """Test error.log rotation"""
        logger = logging.getLogger('test_error_rotation')
        logger.setLevel(logging.INFO)
        
        handler_config = config.config['logging']['handlers']['error_handler']
        test_config = config.config['logging']['files']['error']
        
        max_bytes, backup_count = self.setup_handler(
            logger, handler_config, test_config, log_dir, json_formatter
        )
        
        self.write_test_logs(logger, max_bytes, backup_count)
        self.verify_log_rotation(log_dir, test_config['filename'], 
                               max_bytes, backup_count) 