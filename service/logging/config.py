"""
Logging configuration management for Talkware
"""

import os
import re
import yaml
import logging.config
from typing import Dict, Any, Optional
from datetime import datetime

class LogConfig:
    """
    Logging configuration manager that handles:
    - YAML configuration loading
    - Dynamic log directory creation
    - Environment-specific settings
    """
    
    # Size units in bytes
    SIZE_UNITS = {
        'B': 1,
        'K': 1024,
        'M': 1024 * 1024,
        'G': 1024 * 1024 * 1024
    }
    
    def __init__(self, config_path: str):
        """
        Initialize the logging configuration.
        
        Args:
            config_path (str): Path to logging.yml configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.environment = os.getenv('TALKWARE_ENV', 'development')
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load and parse the YAML configuration file.
        
        Returns:
            dict: Parsed configuration
        """
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        return self.config
        
    def setup_log_directory(self) -> None:
        """
        Create necessary log directories if they don't exist.
        """
        log_dirs = [
            self.config['logging']['directory']['base'],
            self.config['logging']['directory']['archive'],
            self.config['logging']['directory']['temp']
        ]
        
        for directory in log_dirs:
            if directory.startswith('${'):
                # Resolve variable references
                var_name = directory[2:-1]  # Remove ${ and }
                directory = os.getenv(var_name.upper()) or self.config.get(var_name)
                
            if directory:
                os.makedirs(directory, exist_ok=True)
                
    def configure_logging(self) -> None:
        """
        Configure the logging system based on the loaded configuration.
        """
        if not self.config:
            self.load_config()
            
        self.setup_log_directory()
        
        # Create logging configuration dictionary
        log_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'json': {
                    '()': 'libs.logging.CustomJSONFormatter'
                }
            },
            'handlers': {},
            'loggers': {},
            'root': {
                'level': self.config['logging'].get('level', 'INFO'),
                'handlers': []
            }
        }
        
        # Configure handlers for each log file
        for log_type, file_config in self.config['logging']['files'].items():
            handler_name = f'{log_type}_handler'
            log_path = os.path.join(
                self.config['logging']['directory']['base'],
                file_config['filename']
            )
            
            # Add handler configuration
            log_config['handlers'][handler_name] = {
                '()': 'libs.logging.CustomRotatingFileHandler',
                'filename': log_path,
                'maxBytes': self._parse_size(file_config['max_size']),
                'backupCount': file_config['backup_count'],
                'compress': file_config.get('compress', True),
                'formatter': 'json'
            }
            
            # Add logger configuration
            log_config['loggers'][f'talkware.{log_type}'] = {
                'level': 'INFO',
                'handlers': [handler_name],
                'propagate': False
            }
            
            # Add handler to root logger
            if log_type == 'app':
                log_config['root']['handlers'].append(handler_name)
        
        # Apply environment-specific settings
        env_config = self.config.get('environments', {}).get(self.environment, {})
        if env_config:
            self._merge_config(log_config, env_config)
            
        # Configure logging
        logging.config.dictConfig(log_config)
        
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Recursively merge override configuration into base configuration.
        
        Args:
            base (dict): Base configuration
            override (dict): Override configuration
        """
        for key, value in override.items():
            if isinstance(value, dict) and key in base:
                self._merge_config(base[key], value)
            else:
                base[key] = value
                
    def get_log_file_path(self, log_type: str) -> Optional[str]:
        """
        Get the full path for a specific log file.
        
        Args:
            log_type (str): Type of log file (app, inference, access, error)
            
        Returns:
            str: Full path to the log file
        """
        if not self.config:
            self.load_config()
            
        try:
            base_dir = self.config['logging']['directory']['base']
            filename = self.config['logging']['files'][log_type]['filename']
            
            if base_dir.startswith('${'):
                var_name = base_dir[2:-1]
                base_dir = os.getenv(var_name.upper()) or self.config.get(var_name)
                
            return os.path.join(base_dir, filename)
        except KeyError:
            return None
            
    def get_retention_policy(self, log_type: str) -> Dict[str, int]:
        """
        Get retention policy for a specific log type.
        
        Args:
            log_type (str): Type of log file
            
        Returns:
            dict: Retention policy settings
        """
        if not self.config:
            self.load_config()
            
        try:
            return self.config['logging']['retention']['policies'][log_type]
        except KeyError:
            return {'days': 30, 'archive_days': 90}  # Default policy
            
    def _parse_size(self, size_str: str) -> int:
        """
        Parse size string (e.g., '100M') to bytes.
        
        Args:
            size_str (str): Size string with unit (B, K, M, G)
            
        Returns:
            int: Size in bytes
        """
        match = re.match(r'^(\d+)([BKMG])$', size_str.strip())
        if not match:
            raise ValueError(f"Invalid size format: {size_str}. Expected format: NUMBER[B|K|M|G]")
            
        size, unit = match.groups()
        return int(size) * self.SIZE_UNITS[unit] 