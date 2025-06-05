"""
Base model loader interface
"""

import os
import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Literal
import logging
import time

class ProgressLogger:
    def __init__(self, logger, total_shards=4):
        self.logger = logger
        self.total_shards = total_shards
        self.current_shard = 0
        self.start_time = time.time()
        self.last_update = self.start_time

    def update(self, current):
        self.current_shard = current
        current_time = time.time()
        elapsed = current_time - self.start_time
        if current_time - self.last_update >= 1.0:  # 1초마다 업데이트
            progress = (current / self.total_shards) * 100
            speed = current / elapsed if elapsed > 0 else 0
            eta = (self.total_shards - current) / speed if speed > 0 else 0
            bar = "█" * int(progress/2) + "░" * (50 - int(progress/2))
            self.logger.info(f"Loading checkpoint shards: {progress:3.0f}%|{bar}| {current}/{self.total_shards} [{elapsed:.0f}s<{eta:.0f}s, {speed:.2f}it/s]")
            self.last_update = current_time

    def finish(self):
        elapsed = time.time() - self.start_time
        speed = self.total_shards / elapsed if elapsed > 0 else 0
        bar = "█" * 50
        self.logger.info(f"Loading checkpoint shards: 100%|{bar}| {self.total_shards}/{self.total_shards} [{elapsed:.0f}s<00:00, {speed:.2f}it/s]")

class BaseModelLoader(ABC):
    """모델 로더의 기본 인터페이스를 정의하는 추상 클래스"""
    
    def __init__(self, inference_mode: Literal["pipeline", "tokenize"] = "pipeline"):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = None
        self.model_config = None
        self.inference_mode = inference_mode
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        """모델 로드 여부를 반환합니다."""
        return self._is_loaded

    @is_loaded.setter
    def is_loaded(self, value: bool):
        self._is_loaded = value
    
    @abstractmethod
    async def load(self) -> None:
        """모델을 로드합니다."""
        pass
    
    @abstractmethod
    async def unload(self) -> None:
        """모델을 언로드합니다."""
        if self.inference_mode == "pipeline":
            if self.pipeline:
                del self.pipeline
                self.pipeline = None
        else:
            if self.model:
                del self.model
                self.model = None
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @abstractmethod
    async def generate(self,
                      prompt: str,
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      top_p: Optional[float] = None,
                      repetition_penalty: Optional[float] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """텍스트를 생성합니다.
        
        Args:
            prompt: 입력 텍스트
            max_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도
            top_p: 누적 확률 임계값
            repetition_penalty: 반복 패널티
            stop_sequences: 생성 중단 시퀀스 목록
            **kwargs: 추가 키워드 인자
            
        Returns:
            생성된 텍스트
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다."""
        base_info = {
            'inference_mode': self.inference_mode,
            'device': self.device
        }
        return base_info
    
    def _validate_model_path(self, model_path: str) -> None:
        """모델 경로가 유효한지 검증합니다."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
    
    def _setup_device(self) -> None:
        """GPU 사용 가능 여부를 확인하고 device를 설정합니다."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 