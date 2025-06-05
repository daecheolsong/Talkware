"""
Test logging configuration and directory creation
"""

import os
import sys
import pytest
import shutil
import logging
from pathlib import Path
import json

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from service.logging.config import LogConfig
from service.logging.handlers import CustomRotatingFileHandler
from service.logging.formatters import CustomJSONFormatter

class TestLoggingConfig:
    @pytest.fixture
    def app_log_dir(self):
        """Set up the actual log directory"""
        log_dir = os.path.join(project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    @pytest.fixture
    def sample_config(self, clean_log_dir):
        """Create a sample application.yml file with actual log directory"""
        config_dir = os.path.join(project_root, 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        config_content = f"""
# Application Configuration
app:
  name: talkware
  version: 0.1.0
  environment: test
  base_dir: {project_root}

logging:
  version: 1
  disable_existing_loggers: false
  formatters:
    json:
      class: service.logging.formatters.CustomJSONFormatter
  handlers:
    app_handler:
      class: service.logging.handlers.CustomRotatingFileHandler
      formatter: json
      filename: {clean_log_dir}/app.log
      maxBytes: 1024
      backupCount: 3
      compress: true
  root:
    level: INFO
    handlers: [app_handler]
  directory:
    base: {clean_log_dir}
    archive: {clean_log_dir}/archive
    temp: {clean_log_dir}/temp
  files:
    app:
      filename: app.log
      max_size: 100M
      backup_count: 10
      compress: true
    inference:
      filename: inference.log
      max_size: 200M
      backup_count: 20
      compress: true
    access:
      filename: access.log
      max_size: 100M
      backup_count: 10
      compress: true
    error:
      filename: error.log
      max_size: 100M
      backup_count: 30
      compress: true
"""
        config_path = os.path.join(config_dir, 'test_application.yml')
        with open(config_path, 'w') as f:
            f.write(config_content)
        return config_path

    def test_config_loading(self, sample_config):
        """Test if configuration file is loaded correctly"""
        log_config = LogConfig(sample_config)
        config = log_config.load_config()
        
        assert config['app']['name'] == 'talkware'
        assert config['app']['version'] == '0.1.0'
        assert 'logging' in config
        assert 'directory' in config['logging']
        
    def test_log_directory_creation(self, sample_config, clean_log_dir):
        """Test if log directories are created correctly"""
        log_config = LogConfig(sample_config)
        log_config.load_config()
        log_config.setup_log_directory()
        
        # Check if directories are created
        expected_dirs = [
            clean_log_dir,
            os.path.join(clean_log_dir, 'archive'),
            os.path.join(clean_log_dir, 'temp')
        ]
        
        for directory in expected_dirs:
            assert os.path.exists(directory), f"Directory {directory} was not created"
            
    def test_log_file_paths(self, sample_config, clean_log_dir):
        """Test if log file paths are generated correctly"""
        log_config = LogConfig(sample_config)
        log_config.load_config()
        log_config.setup_log_directory()
        
        # Test each log type
        log_types = ['app', 'inference', 'access', 'error']
        for log_type in log_types:
            path = log_config.get_log_file_path(log_type)
            assert path is not None
            assert path.endswith(f'{log_type}.log')
            assert os.path.dirname(path) == clean_log_dir
            
    def test_environment_specific_config(self, sample_config):
        """Test if environment-specific configuration is applied"""
        # Set test environment
        os.environ['TALKWARE_ENV'] = 'test'
        
        log_config = LogConfig(sample_config)
        log_config.load_config()
        
        assert log_config.environment == 'test'
        
    def test_retention_policy(self, sample_config):
        """Test if retention policies are retrieved correctly"""
        log_config = LogConfig(sample_config)
        log_config.load_config()
        
        # Test default policy for non-existent log type
        policy = log_config.get_retention_policy('non_existent')
        assert policy == {'days': 30, 'archive_days': 90}

    def test_actual_logging(self, sample_config, clean_log_dir):
        """Test actual log file creation and writing"""
        # Configure logging
        log_config = LogConfig(sample_config)
        log_config.load_config()
        log_config.setup_log_directory()
        log_config.configure_logging()
        
        # Set up logger
        logger = logging.getLogger('test_logger')
        logger.setLevel(logging.INFO)
        
        # Create and add custom handler
        log_file = os.path.join(clean_log_dir, 'test.log')
        handler = CustomRotatingFileHandler(
            filename=log_file,
            maxBytes=1024,  # 1KB for testing
            backupCount=3,
            compress=True,
            use_timestamp=True
        )
        
        # Add custom formatter
        formatter = CustomJSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Write some test logs
        test_messages = [
            "Test log message 1",
            "Test log message 2",
            "Test log message 3"
        ]
        
        for msg in test_messages:
            logger.info(msg)
            
        # Verify log file exists and contains content
        assert os.path.exists(log_file), f"Log file {log_file} was not created"
        
        with open(log_file, 'r') as f:
            log_contents = f.readlines()
            
        # Verify each message was logged
        assert len(log_contents) == len(test_messages)
        for line in log_contents:
            assert '"level": "INFO"' in line
            assert '"logger": "test_logger"' in line

    def test_application_logging(self, sample_config, clean_log_dir):
        """Test creation and writing to all application log files"""
        # Configure logging
        log_config = LogConfig(sample_config)
        log_config.load_config()
        log_config.setup_log_directory()
        log_config.configure_logging()

        # Write test logs to each logger
        test_messages = {
            'app': ['Application started', 'Configuration loaded'],
            'inference': ['Model loaded', 'Inference completed in 1.2s'],
            'access': ['GET /api/v1/predict', 'POST /api/v1/train'],
            'error': ['Failed to load model', 'Database connection error']
        }
        
        for log_type, messages in test_messages.items():
            logger = logging.getLogger(f'talkware.{log_type}')
            for msg in messages:
                logger.info(msg)
                
        # Verify all log files exist and contain the correct messages
        for log_type, messages in test_messages.items():
            log_file = os.path.join(clean_log_dir, f'{log_type}.log')
            assert os.path.exists(log_file), f"Log file {log_file} was not created"
            
            with open(log_file, 'r') as f:
                log_contents = f.readlines()
                
            # Verify number of messages
            assert len(log_contents) == len(messages), f"Expected {len(messages)} messages in {log_type}.log, but got {len(log_contents)}"
            
            # Verify log format and content
            for line in log_contents:
                log_entry = json.loads(line)
                assert log_entry['level'] == 'INFO'
                assert log_entry['logger'] == f'talkware.{log_type}'
                assert any(msg in log_entry['message'] for msg in messages) 