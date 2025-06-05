"""
Flask 기반의 웹 서버 구현
"""

import os
import logging
from typing import Dict, Any, Optional
from flask import Flask, Blueprint, request, Response
from werkzeug.serving import make_server
import threading
from datetime import datetime
from service.logging import app_logger


class WebServer:
    def __init__(self, config: Dict[str, Any], loggers: Dict[str, logging.Logger]):
        """
        Args:
            config: 서버 설정 (host, port 등)
            loggers: 로거 딕셔너리
        """
        self.config = config
        self.loggers = loggers
        self.app = Flask(__name__)
        self.server = None
        self._server_thread = None
        self.is_running = False
        
        # 로깅 설정
        self.app_logger = loggers['app']
        self.access_logger = loggers['access']
        self.error_logger = loggers['error']
        
        # 서버 설정
        self.host = config.get('server', {}).get('host', '0.0.0.0')
        self.port = int(config.get('server', {}).get('port', 5000))
        
        # CORS 설정
        self._setup_cors()
        
        # 에러 핸들러 등록
        self._register_error_handlers()
        
        # 요청 로깅 설정
        self._setup_request_logging()
        
    def _setup_cors(self):
        """CORS 설정"""
        from flask_cors import CORS
        CORS(self.app)
        
    def _setup_request_logging(self):
        """요청 로깅 설정"""
        @self.app.before_request
        def log_request_start():
            # 요청 시작 시간 저장
            request.start_time = datetime.now()
            
            # HTTPS 요청이 HTTP 서버로 들어올 때 처리
            if request.environ.get('RAW_URI', '').startswith('\x16'):
                self.access_logger.warning(
                    "Received HTTPS request on HTTP port",
                    extra={
                        'remote_addr': request.remote_addr,
                        'message': 'Client attempted HTTPS connection on HTTP port'
                    }
                )
                return 'HTTPS not supported on this port', 400
                
        @self.app.after_request
        def log_request(response: Response):
            # 요청 처리 시간 계산
            duration_ms = None
            if hasattr(request, 'start_time'):
                duration_ms = (datetime.now() - request.start_time).total_seconds() * 1000
                
            # 접근 로그 기록
            self.access_logger.info(
                f"{request.remote_addr} - {request.method} {request.path}",
                extra={
                    'remote_addr': request.remote_addr,
                    'method': request.method,
                    'path': request.path,
                    'query_string': request.query_string.decode('utf-8'),
                    'status_code': response.status_code,
                    'content_length': response.content_length,
                    'duration_ms': duration_ms,
                    'user_agent': request.headers.get('User-Agent'),
                    'referer': request.headers.get('Referer')
                }
            )
            return response
                
    def _register_error_handlers(self):
        """전역 에러 핸들러 등록"""
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            # 에러 로깅
            self.error_logger.error(
                "Unhandled exception",
                exc_info=True,
                extra={'error': str(e)}
            )
            # 에러 응답
            return {
                'error': str(e),
                'type': e.__class__.__name__
            }, 500

    def register_blueprint(self, blueprint: Blueprint, url_prefix: Optional[str] = None):
        """Blueprint 등록
        
        Args:
            blueprint: Flask Blueprint 인스턴스
            url_prefix: URL 접두사 (예: '/api/v1')
        """
        self.app.register_blueprint(blueprint, url_prefix=url_prefix)
        
    def start(self):
        """웹 서버 시작 (비동기)"""
        if self.is_running:
            return
            
        try:
            # 서버 인스턴스 생성
            self.server = make_server(
                host=self.host,
                port=self.port,
                app=self.app,
                threaded=True
            )
            
            # 서버 스레드 시작
            self._server_thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True
            )
            self._server_thread.start()
            
            self.is_running = True
            self.app_logger.info(
                f"Web server started on http://{self.host}:{self.port}",
                extra={
                    'host': self.host,
                    'port': self.port
                }
            )
            
        except Exception as e:
            self.error_logger.error(
                "Failed to start web server",
                exc_info=True,
                extra={
                    'host': self.host,
                    'port': self.port,
                    'error': str(e)
                }
            )
            raise
            
    def stop(self):
        """웹 서버 중지"""
        if not self.is_running:
            return
            
        try:
            # 서버 중지
            if self.server:
                self.server.shutdown()
                self.server = None
            
            # 스레드 종료 대기
            if self._server_thread:
                self._server_thread.join(timeout=5.0)
                self._server_thread = None
                
            self.is_running = False
            self.app_logger.info("Web server stopped")
            
        except Exception as e:
            self.error_logger.error(
                "Error stopping web server",
                exc_info=True,
                extra={'error': str(e)}
            )
            raise 