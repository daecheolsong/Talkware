"""
Test configuration loading and validation from application.yml
"""

import os
import sys
import pytest
import yaml
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from service.logging.config import LogConfig

class TestConfiguration:
    @pytest.fixture
    def config_path(self):
        """Return path to application.yml"""
        return os.path.join(project_root, 'config', 'application.yml')
    
    def test_config_file_exists(self, config_path):
        """Test if configuration file exists"""
        assert os.path.exists(config_path), f"Configuration file not found at {config_path}"
    
    def test_load_config(self, config_path):
        """Test loading configuration file"""
        config = LogConfig(config_path)
        config.load_config()
        assert config.config is not None, "Failed to load configuration"
        
        # Print loaded configuration
        print("\nLoaded configuration:")
        print(yaml.dump(config.config, default_flow_style=False))
    
    def test_app_structure(self, config_path):
        """Test application configuration structure"""
        config = LogConfig(config_path)
        config.load_config()
        
        # Check app section exists
        assert 'app' in config.config, "No app section in config"
        app_config = config.config['app']
        
        # Check required app fields
        required_fields = ['name', 'version', 'environment', 'base_dir']
        for field in required_fields:
            assert field in app_config, f"No {field} in app config"
    
    def test_logging_structure(self, config_path):
        """Test logging configuration structure"""
        config = LogConfig(config_path)
        config.load_config()
        
        # Check logging section exists
        assert 'logging' in config.config, "No logging section in config"
        logging_config = config.config['logging']
        
        # Check required logging fields
        required_fields = ['level', 'format', 'dir', 'file', 'max_size', 'backup_count', 'compress']
        for field in required_fields:
            assert field in logging_config, f"No {field} in logging config"
            
        # Print logging configuration
        print("\nLogging Configuration:")
        for field in required_fields:
            print(f"{field}: {logging_config[field]}")
    
    def test_logging_values(self, config_path):
        """Test logging configuration values"""
        config = LogConfig(config_path)
        config.load_config()
        
        logging_config = config.config['logging']
        
        # Verify level is valid
        assert logging_config['level'] in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        # Verify format is valid
        assert logging_config['format'] in ['json', 'text']
        
        # Verify directory path
        assert '${app.base_dir}' in logging_config['dir']
        
        # Verify file name
        assert logging_config['file'].endswith('.log')
        
        # Verify max_size format (should end with MB)
        assert logging_config['max_size'].endswith('MB')
        size_value = int(logging_config['max_size'][:-2])
        assert 0 < size_value <= 1000, "max_size should be between 1MB and 1000MB"
        
        # Verify backup_count is reasonable
        assert isinstance(logging_config['backup_count'], int)
        assert 0 < logging_config['backup_count'] <= 100
        
        # Verify compress is boolean
        assert isinstance(logging_config['compress'], bool)
        
        print("\nLogging Values Validation:")
        print(f"Level: {logging_config['level']}")
        print(f"Format: {logging_config['format']}")
        print(f"Directory: {logging_config['dir']}")
        print(f"File: {logging_config['file']}")
        print(f"Max Size: {logging_config['max_size']}")
        print(f"Backup Count: {logging_config['backup_count']}")
        print(f"Compress: {logging_config['compress']}") 