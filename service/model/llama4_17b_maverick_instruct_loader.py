"""
Llama4 model loader implementation
"""

import os
import torch
import logging
from service.logging import app_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import hf_hub_download
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
import psutil
from .base_model_loader import BaseModelLoader, ProgressLogger
from .config_loader import ModelConfigLoader


class Llama4MaverickLoader(BaseModelLoader):
    MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct"
    MODEL_ID = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'

    def __init__(self, 
                 config_loader: ModelConfigLoader, 
                 loggers: Dict[str, logging.Logger],
                 inference_mode: Literal["pipeline", "tokenize"] = "pipeline",
                 max_new_tokens: int = 512):
        """
        Args:
            config_loader: 모델 설정 로더 인스턴스
            loggers: 로거 딕셔너리 {'app': app_logger, 'inference': inference_logger, ...}
            inference_mode: 추론 방식 선택 ("pipeline" 또는 "tokenize")
            max_new_tokens: 최대 생성 토큰 수
        """
        super().__init__(inference_mode=inference_mode)
        self.model_name = self.MODEL_NAME
        self.config_loader = config_loader
        self.loggers = loggers
        self.inference_mode = inference_mode
        self.max_new_tokens = max_new_tokens
        self.model_id = self.MODEL_ID
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map = "auto"
        self._is_loaded = False
        
        # 로깅 설정
        self.app_logger = loggers['app']
        self.inference_logger = loggers['inference']
        self.error_logger = loggers['error']

    @property
    def is_loaded(self) -> bool:
        """모델 로드 여부를 반환합니다."""
        return self._is_loaded

    @is_loaded.setter
    def is_loaded(self, value: bool):
        self._is_loaded = value

    async def load(self) -> None:
        """비동기적으로 모델을 로드합니다."""
        try:
            self.app_logger.info(f"Loading {self.model_name} model using {self.inference_mode} mode...", 
                               extra={'device': self.device})

            # 초기 GPU 메모리 사용량 기록
            if self.device == "cuda":
                initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB 단위
                self.app_logger.info(f"Initial GPU memory usage: {initial_memory:.2f}MB")

            # 캐시 디렉토리 설정
            os.environ['TRANSFORMERS_CACHE'] = self.config_loader.hf_hub_cache

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.config_loader.hf_hub_cache,
                token=self.config_loader.hf_token,
                trust_remote_code=True
            )

            # (2) A100 80GB 카드 4장에 대해 모델용으로 72GB씩 할당하도록 max_memory dict 정의
            max_memory = {
                0: "72GB",
                1: "72GB",
                2: "72GB",
                3: "72GB"
            }

            # 모델 로드 - 진행 상태를 직접 모니터링
            self.app_logger.info("Starting model loading with checkpoint shards...")
            progress_logger = ProgressLogger(self.app_logger)
            
            # 모델 로드 시작
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                cache_dir=self.config_loader.hf_hub_cache,
                token=self.config_loader.hf_token,
                device_map=self.device_map,  # 기본값: 자동으로 오프로딩까지 감안
                max_memory=max_memory,
                low_cpu_mem_usage=True, # 메모리 사용량을 줄이기 위해 사용
                offload_folder=None, # offload_folder를 None으로 설정하면 CPU로 처리하지 않음
                offload_state_dict=False # 파라미터 체크포인트도 CPU로 내리지 않음
            )
            
            # 모델 로드 완료 후 진행 상태 업데이트
            progress_logger.finish()
            self.app_logger.info("Model loading completed")

            if self.inference_mode == "pipeline":
                from transformers import pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device_map=self.device_map,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            self.is_loaded = True

            # 최종 GPU 메모리 사용량 계산 및 로깅
            if self.device == "cuda":
                final_memory = torch.cuda.memory_allocated() / 1024**2
                memory_used = final_memory - initial_memory
                
                self.inference_logger.info(
                    f"Model {self.model_name} loaded successfully on {self.device}. "
                    f"GPU Memory used: {memory_used:.2f}MB"
                )
            else:
                self.inference_logger.info(
                    f"Model {self.model_name} loaded successfully on {self.device}"
                )
            
            self.app_logger.info(f"Model {self.model_name} initialization completed")

        except Exception as e:
            self.error_logger.error(
                f"Failed to load model {self.model_name}: {str(e)}",
                exc_info=True
            )
            raise

    async def unload(self) -> None:
        """모델을 언로드하고 리소스를 정리합니다."""
        try:
            if self.is_loaded:
                # GPU 메모리 초기 상태 기록
                if torch.cuda.is_available():
                    initial_memory = torch.cuda.memory_allocated() / 1024**2

                # 모델 메모리 해제
                if self.inference_mode == "pipeline":
                    if self.pipeline is not None:
                        del self.pipeline
                        self.pipeline = None
                else:
                    if self.model is not None:
                        del self.model
                        self.model = None
                    if self.tokenizer is not None:
                        del self.tokenizer
                        self.tokenizer = None

                # CUDA 캐시 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    final_memory = torch.cuda.memory_allocated() / 1024**2
                    memory_freed = initial_memory - final_memory
                    
                    self.inference_logger.info(
                        "Model unloaded successfully",
                        extra={
                            'model_name': self.model_name,
                            'gpu_memory_freed_mb': memory_freed
                        }
                    )

                self.is_loaded = False
                self.app_logger.info(f"{self.model_name} unloaded successfully")

        except Exception as e:
            self.error_logger.error(
                "Failed to unload model",
                exc_info=True,
                extra={
                    'model_name': self.model_name,
                    'error': str(e)
                }
            )
            raise

    async def generate(self,
                      prompt: str,
                      max_tokens: Optional[int] = 2048,
                      temperature: Optional[float] = 0.7,
                      top_p: Optional[float] = 0.9,
                      repetition_penalty: Optional[float] = 1.1,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """텍스트 생성"""
        try:
            start_time = datetime.now()
            request_id = kwargs.get('request_id', '')
            
            # 입력 크기 계산
            input_size = len(prompt.encode('utf-8'))
            
            # 접근 로그 기록
            self.loggers['access'].info(
                "Text generation request received",
                extra={
                    'request_id': request_id,
                    'model_name': self.model_name,
                    'input_size': input_size,
                    'max_tokens': max_tokens
                }
            )
            
            # GPU 메모리 사용량 체크
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB 단위
            else:
                gpu_memory = 0
                
            if self.inference_mode == "pipeline":
                # Pipeline을 사용한 텍스트 생성
                generation_config = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": True,
                    "pad_token_id": self.pipeline.tokenizer.pad_token_id,
                    "eos_token_id": self.pipeline.tokenizer.eos_token_id,
                }
                
                if stop_sequences:
                    generation_config["stopping_criteria"] = stop_sequences

                # 텍스트 생성 실행
                output = self.pipeline(
                    prompt,
                    **generation_config
                )
                
                generated_text = output[0]['generated_text']
            
            else:  # tokenize 모드
                # 입력 텍스트 토크나이징
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 생성 설정
                generation_config = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }

                if stop_sequences:
                    generation_config["stopping_criteria"] = stop_sequences

                # 텍스트 생성
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, **generation_config)
                
                # 생성된 텍스트 디코딩
                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # 출력 크기 계산
            output_size = len(generated_text.encode('utf-8'))
            
            # 추론 시간 계산
            inference_time = (datetime.now() - start_time).total_seconds() * 1000  # ms 단위
            
            # 추론 결과 로깅
            self.loggers['inference'].info(
                "Text generation completed",
                extra={
                    'request_id': request_id,
                    'model_name': self.model_name,
                    'model_version': '17b',
                    'inference_time': inference_time,
                    'input_size': input_size,
                    'output_size': output_size,
                    'memory_usage': psutil.Process().memory_info().rss / (1024 * 1024),
                    'gpu_usage': gpu_memory,
                    'inference_mode': self.inference_mode
                }
            )
            
            return generated_text
            
        except Exception as e:
            # 에러 로그에 기록
            self.loggers['error'].error(
                "Text generation failed",
                exc_info=True,
                extra={
                    'request_id': request_id,
                    'model_name': self.model_name,
                    'error': str(e),
                    'prompt_length': len(prompt),
                    'inference_mode': self.inference_mode
                }
            )
            # 앱 로그에도 기록
            self.loggers['app'].error(
                f"Text generation failed for {self.model_name}",
                extra={
                    'request_id': request_id,
                    'model_name': self.model_name,
                    'error': str(e)
                }
            )
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다."""
        info = super().get_model_info()
        info.update({
            'model_name': self.model_name,
            'model_type': 'llama4',
            'model_version': '17b',
            'is_loaded': self.is_loaded,
        })
        
        if torch.cuda.is_available():
            info.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2
            })
            
        return info 