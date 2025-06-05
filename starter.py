#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import signal
import asyncio
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from service.model import (
    ModelConfigLoader, Llama3Loader, Llama3QuantizedLoader,
    Llama4ScoutLoader, Llama4MaverickLoader, Llama32BLoader
)
from service.logging import app_logger
from service.web.server import WebServer
from controller.inference_controller import inference_bp, init_controller

print("Starting application...")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")


class ConfigLoader:
    """
    애플리케이션 설정(application.yml 및 imports) 파일을 로드하고 병합하는 클래스
    - 환경 변수 참조(${VAR:default})를 실제 값으로 치환
    - imports 섹션에 지정된 추가 YAML 파일을 재귀적으로 병합
    """
    def __init__(self, base_dir: str = None):
        # base_dir이 주어지지 않으면 스크립트 위치(parent)로 설정
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        # 설정 파일 디렉토리 경로
        self.config_dir = self.base_dir / 'config'
        # 최종 병합된 설정을 저장할 딕셔너리
        self.config: Dict[str, Any] = {}

    def load_config(self) -> Dict[str, Any]:
        """설정 로딩 진입점: application.yml → 환경 변수 처리 → imports 순으로 적용"""
        try:
            # 1) 메인 설정 파일 로드
            main_config = self._load_yaml('application.yml')
            self.config.update(main_config)

            # 2) 설정 전체에 환경 변수 치환 적용
            self._process_env_vars(self.config)

            # 3) imports가 있으면 추가 파일을 로드하여 재귀 병합
            for import_file in main_config.get('imports', []):
                imported = self._load_yaml(import_file)
                self._process_env_vars(imported)
                self._deep_update(self.config, imported)

            return self.config
        except Exception as e:
            print(f"설정 로드 중 오류 발생: {e}")
            raise

    def _deep_update(self, base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
        """딕셔너리를 깊이(재귀) 병합합니다."""
        for key, val in upd.items():
            if isinstance(val, dict):
                base[key] = self._deep_update(base.get(key, {}), val)
            else:
                base[key] = val
        return base

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """단일 YAML 파일을 안전하게 로드합니다."""
        path = self.config_dir / filename
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"'{filename}' 파일 로드 중 오류 발생: {e}")
            raise

    def _process_env_vars(self, cfg: Dict[str, Any], parent: str = '') -> None:
        """
        '${VAR:default}' 또는 '${app.base_dir}' 형태를 실제 값으로 치환합니다.
        - app.base_dir: 메인 설정의 app.base_dir 값 또는 스크립트 위치
        """
        for key, val in cfg.items():
            if isinstance(val, dict):
                self._process_env_vars(val, f"{parent}.{key}" if parent else key)
            elif isinstance(val, str) and '${' in val:
                result = val
                # 여러 env 치환 지원
                while '${' in result and '}' in result:
                    start = result.find('${')
                    end = result.find('}', start)
                    if end == -1:
                        break
                    var_def = result[start+2:end]
                    if ':' in var_def:
                        name, default = var_def.split(':', 1)
                    else:
                        name, default = var_def, None

                    # app.base_dir 특수 처리
                    if name == 'app.base_dir':
                        repl = self.config.get('app', {}).get('base_dir', str(self.base_dir))
                    else:
                        repl = os.environ.get(name, default) or str(self.base_dir)

                    result = result.replace(f"${{{var_def}}}", str(repl))
                cfg[key] = result


# 전역 변수: 시그널 수신 시 True로 변경되어 메인 루프를 종료시킵니다.
is_shutting_down = False

def signal_handler(signum, frame):
    """SIGTERM, SIGINT 같은 종료 시그널이 오면 호출됩니다."""
    global is_shutting_down
    is_shutting_down = True


def setup_logging(config: Dict[str, Any]) -> None:
    """로깅 시스템 초기화
    
    Args:
        config: 애플리케이션 설정
    """
    try:
        print("Setting up logging...")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Base directory: {config['app'].get('base_dir')}")
        
        # app 설정을 먼저 전달
        app_logger.app_config = config['app']
        
        # YAML 설정 파일 로드
        log_config_path = str(Path(config['app']['base_dir']) / 'config' / 'logging.yml')
        print(f"Loading logging config from: {log_config_path}")
        
        with open(log_config_path, 'r', encoding='utf-8') as f:
            log_config = yaml.safe_load(f)
            
        # base_dir 변수 해석
        base_dir = config['app']['base_dir']
        for key in ['log_dir', 'archive_dir', 'temp_dir']:
            if key in log_config:
                log_config[key] = log_config[key].replace('${app.base_dir}', base_dir)
        
        for handler in log_config.get('handlers', {}).values():
            if 'filename' in handler:
                handler['filename'] = handler['filename'].replace('${app.base_dir}', base_dir)
        
        # 로깅 설정 적용
        app_logger.setup(log_config)
        print("Logging setup completed")
        
        # 로그 파일 경로 확인
        log_dir = os.path.join(base_dir, 'logs')
        print(f"Log directory: {log_dir}")
        if os.path.exists(log_dir):
            print(f"Log files in directory: {os.listdir(log_dir)}")
        else:
            print(f"Log directory does not exist: {log_dir}")
            
    except Exception as e:
        print(f"로깅 설정 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """애플리케이션 진입점: 설정 → 로깅 → 모델 로딩 → 서비스 루프"""
    # 종료 시그널 핸들러 등록
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    start_time = datetime.now()
    web_server = None
    model_loader = None
    app_name = 'talkware'
    app_version = '0.1.0'
    
    try:
        # 설정 로딩
        loader = ConfigLoader()
        config = loader.load_config()
        
        # 로깅 초기화
        setup_logging(config)
        app_logger.get_logger('app').info("Logging system initialized")

        # 애플리케이션 정보
        app_name = config['app'].get('name', 'talkware')
        app_version = config['app'].get('version', '0.1.0')

        # 서버 설정 읽기
        server = config.get('server', {})
        host = server.get('host')
        port = server.get('port')

        # 모델 설정 로드
        mcfg = config.get('model', {})
        base_dir = mcfg.get('model_base_dir')
        cache_dir = mcfg.get('hf_hub_cache')
        token_path = mcfg.get('hf_token_path')
        loader_model = mcfg.get('loader_model')
        inference_mode = mcfg.get('inference_mode')
        quantization = mcfg.get('quantization', '4bit')  # 기본값은 4bit

        # ModelConfigLoader 초기화
        model_config_loader = ModelConfigLoader(base_dir, cache_dir, token_path)
        app_logger.get_logger('app').info("Model configuration loaded successfully")

        # 시작 모델 설정 확인
        if not loader_model:
            raise ValueError("loader_model must be specified in configuration")            

        # 모델 타입에 따른 로더 선택
        if loader_model.lower() == "llama3":
            ModelLoaderClass = Llama3Loader
        elif loader_model.lower() == "llama3_quantized":
            ModelLoaderClass = Llama3QuantizedLoader           
        elif loader_model.lower() == "llama32b":
            ModelLoaderClass = Llama32BLoader            
        elif loader_model.lower() == "llama4_scout":
            ModelLoaderClass = Llama4ScoutLoader
        elif loader_model.lower() == "llama4_maverick":
            ModelLoaderClass = Llama4MaverickLoader
        else:
            raise ValueError(f"Unsupported model type: {loader_model}")
        
        app_logger.get_logger('app').info(f"Initializing {loader_model} loader")
        
        # 선택된 모델 로더 초기화 및 로드
        loggers = {
            'app': app_logger.get_logger('app'),         # 애플리케이션 전반적인 상태/이벤트
            'inference': app_logger.get_logger('inference'),  # 모델 추론 관련
            'error': app_logger.get_logger('error'),     # 에러 상황
            'access': app_logger.get_logger('access')    # API 접근/요청 관련
        }
        
        # 모델 로더 초기화 및 로드
        # 모델 로더 클래스 동적 인스턴스 생성
        model_loader_kwargs = {
            'config_loader': model_config_loader,
            'loggers': loggers,
            'inference_mode': inference_mode or "pipeline"  # 기본값은 pipeline 모드
        }

        # Llama3QuantizedLoader인 경우 quantization 파라미터 추가
        if loader_model.lower() == "llama3quantizedloader":
            if quantization not in ['4bit', '8bit']:
                raise ValueError(f"Invalid quantization value for Llama3QuantizedLoader: {quantization}. Must be '4bit' or '8bit'")
            model_loader_kwargs['quantization'] = quantization

        model_loader = ModelLoaderClass(**model_loader_kwargs)

        # 모델 비동기 로드
        asyncio.run(model_loader.load())
        
        app_logger.get_logger('app').info(
            "Model loaded successfully",
            extra=model_loader.get_model_info()
        )

        # 컨트롤러 초기화
        init_controller(model_loader, loggers)
        
        # 웹서버 초기화 및 시작
        web_server = WebServer(config, loggers)
        web_server.register_blueprint(inference_bp, url_prefix='/api/v1')
        web_server.start()

        # 서비스 시작 로그 및 시간 측정
        app_logger.get_logger('app').info(f"Starting {app_name}[{app_version}] server...")
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        app_logger.get_logger('app').info(
            "Server startup complete",
            extra={'startup_time_ms': elapsed, 'host': host, 'port': port}
        )

        # 메인 루프: 종료 플래그 전까지 대기
        while not is_shutting_down:
            import time; time.sleep(1)

    except Exception as e:
        # 에러 로깅
        try:
            app_logger.get_logger('error').error(
                "Application error",
                exc_info=True,
                extra={'error': str(e)}
            )
        except:
            print(f"Error occurred: {e}")
            import traceback; traceback.print_exc()
    
    finally:
        # 종료 처리
        try:
            app_logger.get_logger('app').info(f"Shutting down {app_name}[{app_version}]...")
            
            # 웹서버 종료
            if web_server and web_server.is_running:
                web_server.stop()
            
            # 모델 언로드
            if model_loader and model_loader.is_loaded:
                try:
                    asyncio.run(model_loader.unload())
                    app_logger.get_logger('app').info(f"Model {loader_model} unloaded successfully")
                except Exception as ue:
                    app_logger.get_logger('error').error(
                        "Failed to unload model during shutdown",
                        exc_info=True,
                        extra={'error': str(ue)}
                    )
            
            app_logger.get_logger('app').info("Application Shutdown")
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)
        
        # 정상 종료
        if not is_shutting_down:
            sys.exit(1)  # 에러로 인한 종료
        else:
            sys.exit(0)  # 정상 종료

if __name__ == "__main__":
    main()