"""
Talkware Logging Module

This module provides custom logging functionality for the Talkware application.
It includes custom handlers, formatters, and configuration management.
"""

import logging
import atexit
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import os
import logging.config

from .handlers import CustomRotatingFileHandler
from .formatters import CustomJSONFormatter
from .config import LogConfig

class AppLogger:
    """Application logger setup and management"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the logger"""
        self.initialized = False
        self.log_dir = None
        self.archive_dir = None
        self.temp_dir = None
        self.start_time = datetime.now()
        self.app_config: Dict = {}
    
    def setup(self, config: Dict[str, Any]) -> None:
        """
        Set up logging configuration
        
        Args:
            config: Configuration dictionary from logging.yml
        """
        if self.initialized:
            return
            
        # 로그 디렉토리 설정 - 절대 경로로 변환
        base_dir = self.app_config.get('base_dir', os.getcwd())
        self.log_dir = os.path.abspath(os.path.join(base_dir, config.get('log_dir', 'logs')))
        self.archive_dir = os.path.abspath(os.path.join(base_dir, config.get('archive_dir', os.path.join(self.log_dir, 'archive'))))
        self.temp_dir = os.path.abspath(os.path.join(base_dir, config.get('temp_dir', os.path.join(self.log_dir, 'temp'))))
        
        # 디렉토리 생성
        for directory in [self.log_dir, self.archive_dir, self.temp_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        # 환경 변수 설정
        os.environ['log_dir'] = self.log_dir
        os.environ['archive_dir'] = self.archive_dir
        os.environ['temp_dir'] = self.temp_dir
        
        # config의 경로도 절대 경로로 업데이트
        config['log_dir'] = self.log_dir
        config['archive_dir'] = self.archive_dir
        config['temp_dir'] = self.temp_dir
        
        # 로깅 설정 적용
        logging.config.dictConfig(config)
        
        self.initialized = True
        
        # 종료 시 로깅을 위한 핸들러 등록
        atexit.register(self.log_shutdown)

    def _get_base_extra(self) -> dict:
        """기본 로그 필드를 반환합니다."""
        return {
            'app_name': self.app_config.get('name', 'talkware'),
            'version': self.app_config.get('version', '0.1.0'),
            'environment': self.app_config.get('environment', 'development'),
            'base_dir': str(self.app_config.get('base_dir', os.getcwd()))
        }

    def info(self, message: str, extra: dict = None):
        """INFO 레벨 로그를 기록합니다."""
        log_extra = self._get_base_extra()
        if extra:
            log_extra.update(extra)
        logging.getLogger('app').info(message, extra=log_extra)
    
    def error(self, message: str, extra: dict = None):
        """ERROR 레벨 로그를 기록합니다."""
        log_extra = self._get_base_extra()
        if extra:
            log_extra.update(extra)
        logging.getLogger('app').error(message, extra=log_extra)

    def log_shutdown(self):
        """앱 종료를 로깅합니다."""
        end_time = datetime.now()
        runtime = end_time - self.start_time
        
        self.info("Application Shutdown", extra={
            'end_time': end_time.isoformat(),
            'total_runtime': str(runtime),
            'runtime_seconds': runtime.total_seconds()
        })

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance
        
        Args:
            name: Logger name
            
        Returns:
            logging.Logger: Logger instance
        """
        return logging.getLogger(name)

# 싱글톤 인스턴스
app_logger = AppLogger()

__version__ = '0.1.0'
__all__ = ['app_logger'] 