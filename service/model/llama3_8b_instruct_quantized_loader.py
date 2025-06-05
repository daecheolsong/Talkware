"""
Llama3 quantized model loader implementation (4-bit and 8-bit quantization support)
"""

import os
import torch
import logging
from service.logging import app_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
import psutil
from .base_model_loader import BaseModelLoader, ProgressLogger
from .config_loader import ModelConfigLoader


class Llama3QuantizedLoader:
    MODEL_NAME = "Llama-3.1-8B-Instruct-Quantized"
    MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'

    def __init__(self, 
                 config_loader: ModelConfigLoader, 
                 loggers: Dict[str, logging.Logger],
                 inference_mode: Literal["pipeline", "tokenize"] = "pipeline",
                 quantization: Literal["4bit", "8bit"] = "4bit",
                 max_new_tokens: int = 512):
        """
        Args:
            config_loader: 모델 설정 로더 인스턴스
            loggers: 로거 딕셔너리 {'app': app_logger, 'inference': inference_logger, ...}
            inference_mode: 추론 방식 선택 ("pipeline" 또는 "tokenize")
            quantization: 양자화 방식 선택 ("4bit" 또는 "8bit")
            max_new_tokens: 최대 생성 토큰 수
        """
        self.model_name = self.MODEL_NAME
        self.config_loader = config_loader
        self.loggers = loggers
        self.inference_mode = inference_mode
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        self.model_id = self.MODEL_ID
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map = "auto"
        
        if self.device != "cuda":
            raise ValueError("Quantization is only supported on CUDA devices")
        
        # 로깅 설정
        self.app_logger = loggers['app']
        self.inference_logger = loggers['inference']
        self.error_logger = loggers['error']

    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """양자화 설정을 반환합니다."""
        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:  # 8bit
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )

    async def load(self) -> None:
        """비동기적으로 모델을 로드합니다."""
        try:
            self.app_logger.info(
                f"Loading {self.model_name} model using {self.inference_mode} mode "
                f"with {self.quantization} quantization...", 
                extra={'device': self.device}
            )

            # 초기 GPU 메모리 사용량 기록
            initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB 단위
            self.app_logger.info(f"Initial GPU memory usage: {initial_memory:.2f}MB")

            # 캐시 디렉토리 설정
            os.environ['TRANSFORMERS_CACHE'] = self.config_loader.hf_hub_cache

            # 양자화 설정 가져오기
            quantization_config = self._get_quantization_config()

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.config_loader.hf_hub_cache,
                token=self.config_loader.hf_token,
                trust_remote_code=True,
                use_fast=True,
                local_files_only=True
            )

            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.app_logger.info("Set pad_token to eos_token for quantized model tokenizer")
                else:
                    # EOS 토큰도 없는 경우 새로운 패딩 토큰 추가
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.app_logger.info("Added new pad_token '[PAD]' to quantized model tokenizer")

            # 모델 로드 - 진행 상태를 직접 모니터링
            self.app_logger.info("Starting model loading with checkpoint shards...")
            progress_logger = ProgressLogger(self.app_logger)
            
            # 모델 로드 시작
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=self.config_loader.hf_hub_cache,
                token=self.config_loader.hf_token,
                device_map=self.device_map,
                local_files_only=True,
                quantization_config=quantization_config,
                trust_remote_code=True
            )
            
            # 모델 로드 완료 후 진행 상태 업데이트
            progress_logger.finish()
            self.app_logger.info("Model loading completed")

            # 모델의 패딩 토큰 ID 설정
            if not hasattr(self.model.config, 'pad_token_id') or self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                self.app_logger.info(f"Set quantized model pad_token_id to {self.tokenizer.pad_token_id}")
            else:
                self.app_logger.info(f"Quantized model already has pad_token_id: {self.model.config.pad_token_id}")

            # 모델의 임베딩 레이어 크기 조정 (새로운 토큰이 추가된 경우)
            if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.app_logger.info(f"Resized quantized model embeddings to match tokenizer size: {len(self.tokenizer)}")

            if self.inference_mode == "pipeline":
                # 로드된 모델과 토크나이저로 pipeline 생성
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
            final_memory = torch.cuda.memory_allocated() / 1024**2
            memory_used = final_memory - initial_memory
            
            self.inference_logger.info(
                f"Model {self.model_name} loaded successfully on {self.device}. "
                f"GPU Memory used: {memory_used:.2f}MB "
                f"with {self.quantization} quantization"
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
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated() / 1024**2
                memory_freed = initial_memory - final_memory
                
                self.inference_logger.info(
                    "Model unloaded successfully",
                    extra={
                        'model_name': self.model_name,
                        'gpu_memory_freed_mb': memory_freed,
                        'quantization': self.quantization
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
                    'max_tokens': max_tokens,
                    'quantization': self.quantization
                }
            )
            
            # GPU 메모리 사용량 체크
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB 단위
                
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
                    'model_version': '3.8b',
                    'inference_time': inference_time,
                    'input_size': input_size,
                    'output_size': output_size,
                    'memory_usage': psutil.Process().memory_info().rss / (1024 * 1024),
                    'gpu_usage': gpu_memory,
                    'inference_mode': self.inference_mode,
                    'quantization': self.quantization
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
                    'inference_mode': self.inference_mode,
                    'quantization': self.quantization
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
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'inference_mode': self.inference_mode,
            'quantization': self.quantization
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2
            })
            
        return info 