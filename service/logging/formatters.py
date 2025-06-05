"""
Custom logging formatters for Talkware
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
import time
import threading


class DetailedFormatter(logging.Formatter):
    """상세 정보를 포함하는 로그 포맷터"""
    
    def __init__(self, app_name: str = "talkware"):
        super().__init__()
        self.app_name = app_name
        
    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 포맷팅"""
        # ISO 8601 형식의 타임스탬프 생성
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # 스레드 이름 가져오기
        thread_name = threading.current_thread().name
        
        # 기본 메시지 포맷
        base_format = f"{timestamp}  {record.levelname:5} 1 --- [{self.app_name}] [{thread_name}] "
        
        # 전체 메시지 구성
        message = f"{base_format}{record.getMessage()}"
        
        # 예외 정보가 있으면 추가
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                message = f"{message}\n{record.exc_text}"
                
        # 스택 정보가 있으면 추가
        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"
            
        return message


class KoreanFormatter(logging.Formatter):
    """한글 메시지용 포맷터"""
    
    def __init__(self):
        super().__init__()
        
    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 한글 포맷으로 포맷팅"""
        # 날짜 형식 변환
        timestamp = datetime.fromtimestamp(record.created).strftime("%d-%b-%Y %H:%M:%S.%f")[:-3]
        
        # 로그 레벨 한글화
        level_map = {
            'WARNING': '경고',
            'INFO': '정보',
            'ERROR': '오류',
            'DEBUG': '디버그',
            'CRITICAL': '심각'
        }
        level_name = level_map.get(record.levelname, record.levelname)
        
        # 스레드 이름
        thread_name = threading.current_thread().name
        
        # 메시지 구성
        message = f"{timestamp} {level_name} [{thread_name}] {record.getMessage()}"
        
        # 예외 정보 추가
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                message = f"{message}\n{record.exc_text}"
                
        return message


class BaseJSONFormatter(logging.Formatter):
    """기본 JSON 포맷터"""
    
    def __init__(self, include_fields: Optional[list] = None):
        super().__init__()
        self.include_fields = include_fields or [
            'timestamp', 'level', 'message', 'logger'
        ]
        
    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 JSON으로 포맷팅"""
        message = record.getMessage()
        log_dict: Dict[str, Any] = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': message,
            'logger': record.name
        }
        
        # 추가 필드가 있으면 포함
        if hasattr(record, 'extra_fields'):
            log_dict.update(record.extra_fields)
            
        # 예외 정보가 있으면 포함
        if record.exc_info:
            log_dict['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        # 필드 필터링
        if self.include_fields:
            log_dict = {k: v for k, v in log_dict.items()
                       if k in self.include_fields}
            
        return json.dumps(log_dict, ensure_ascii=False)


class AppLogFormatter(BaseJSONFormatter):
    """애플리케이션 로그 포맷터"""
    
    def __init__(self):
        super().__init__(include_fields=[
            'timestamp', 'level', 'message', 'logger',
            'app_name', 'version', 'environment', 'module',
            'function', 'line'
        ])


class ErrorLogFormatter(BaseJSONFormatter):
    """에러 로그 포맷터"""
    
    def __init__(self):
        super().__init__(include_fields=[
            'timestamp', 'level', 'message', 'logger',
            'error_type', 'error_details', 'traceback',
            'module', 'function', 'line'
        ])


class AccessLogFormatter(BaseJSONFormatter):
    """접근 로그 포맷터"""
    
    def __init__(self):
        super().__init__(include_fields=[
            'timestamp', 'level', 'message', 'logger',
            'request_id', 'method', 'path', 'status_code',
            'response_time', 'client_ip', 'user_agent'
        ])


class InferenceLogFormatter(BaseJSONFormatter):
    """추론 로그 포맷터"""
    
    def __init__(self):
        super().__init__(include_fields=[
            'timestamp', 'level', 'message', 'logger',
            'request_id', 'model_name', 'model_version',
            'inference_time', 'input_size', 'output_size',
            'memory_usage', 'gpu_usage'
        ])


# 이전 버전과의 호환성을 위해 CustomJSONFormatter 유지
CustomJSONFormatter = AppLogFormatter

__all__ = [
    'BaseJSONFormatter',
    'AppLogFormatter',
    'ErrorLogFormatter',
    'AccessLogFormatter',
    'InferenceLogFormatter',
    'CustomJSONFormatter',
    'DetailedFormatter',
    'KoreanFormatter'
] 