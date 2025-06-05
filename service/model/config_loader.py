"""
Configuration loader for model settings
"""

import os
from typing import Dict, Any, Optional


class ModelConfigLoader:
    def __init__(
        self,
        model_base_dir: str,
        hf_hub_cache: str,
        hf_token_path: str
    ):
        """모델 설정 로더 초기화
        
        Args:
            model_base_dir: 모델 파일이 저장되는 기본 디렉토리
            hf_hub_cache: Hugging Face 캐시 디렉토리
            hf_token_path: Hugging Face 토큰 파일 경로
            
        Raises:
            ValueError: 필수 입력값이 None이거나 빈 문자열인 경우
            FileNotFoundError: 토큰 파일이 존재하지 않는 경우
        """
        # 필수 입력값 검증
        if not model_base_dir or not hf_hub_cache or not hf_token_path:
            raise ValueError("All parameters (model_base_dir, hf_hub_cache, hf_token_path) are required")
            
        self.model_base_dir = model_base_dir
        self.hf_hub_cache = hf_hub_cache
        self.hf_token_path = hf_token_path
        
        # 토큰 파일 존재 확인
        if not os.path.exists(hf_token_path):
            raise FileNotFoundError(f"Token file not found: {hf_token_path}")
            
        # 토큰 값 로드
        self.hf_token = self._load_token()
        if not self.hf_token:
            raise ValueError(f"Could not find HF_TOKEN in file: {hf_token_path}")
            
    def _load_token(self) -> Optional[str]:
        """토큰 파일에서 HF_TOKEN 값을 읽음"""
        try:
            with open(self.hf_token_path, 'r') as f:
                for line in f:
                    if line.startswith('HF_TOKEN='):
                        return line.split('=', 1)[1].strip()
            return None
        except Exception as e:
            raise ValueError(f"Failed to read token file: {str(e)}")
            
    def get_model_base_dir(self) -> str:
        """모델 기본 디렉토리 경로 반환"""
        return self.model_base_dir
        
    def get_hf_hub_cache(self) -> str:
        """Hugging Face 캐시 디렉토리 경로 반환"""
        return self.hf_hub_cache
        
    def get_hf_token_path(self) -> str:
        """Hugging Face 토큰 파일 경로 반환"""
        return self.hf_token_path
        
    def get_hf_token(self) -> str:
        """Hugging Face 토큰 값 반환"""
        return self.hf_token

        