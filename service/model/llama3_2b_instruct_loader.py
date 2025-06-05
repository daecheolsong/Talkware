"""
Llama-3.2-3B-Instruct model loader implementation
"""

import os
import torch
import logging
import sys
import time
from service.logging import app_logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
import psutil
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto
from .config_loader import ModelConfigLoader
from .base_model_loader import BaseModelLoader, ProgressLogger

class Llama32BLoader:
    MODEL_NAME = "Llama-3.2-3B-Instruct"
    MODEL_ID = 'meta-llama/Llama-3.2-3B-Instruct'

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
        self.model_name = self.MODEL_NAME
        self.config_loader = config_loader
        self.loggers = loggers
        self.inference_mode = inference_mode
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        self.model_id = self.MODEL_ID
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map = "cuda:0"
        
        # 로깅 설정
        self.app_logger = loggers['app']
        self.inference_logger = loggers['inference']
        self.error_logger = loggers['error']

    async def load(self) -> None:
        """비동기적으로 모델을 로드합니다."""
        try:
            self.app_logger.info(f"Loading {self.model_name} model using {self.inference_mode} mode...", 
                               extra={'device': self.device})

            # 초기 GPU 메모리 사용량 기록
            if self.device == "cuda":
                initial_memory = torch.cuda.memory_allocated() / 1024**2
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

            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.app_logger.info("Set pad_token to eos_token for tokenizer")
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.app_logger.info("Added new pad_token '[PAD]' to tokenizer")

            # 모델 로드
            self.app_logger.info("Starting model loading...")
            progress_logger = ProgressLogger(self.app_logger)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                cache_dir=self.config_loader.hf_hub_cache,
                token=self.config_loader.hf_token,
                device_map=self.device_map,
                low_cpu_mem_usage=True
            )
            
            progress_logger.finish()
            self.app_logger.info("Model loading completed")

            # 모델의 패딩 토큰 ID 설정
            if not hasattr(self.model.config, 'pad_token_id') or self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                self.app_logger.info(f"Set model pad_token_id to {self.tokenizer.pad_token_id}")
            else:
                self.app_logger.info(f"Model already has pad_token_id: {self.model.config.pad_token_id}")

            # 모델의 임베딩 레이어 크기 조정
            if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.app_logger.info(f"Resized model embeddings to match tokenizer size: {len(self.tokenizer)}")

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
        if self.is_loaded:
            try:
                if self.pipeline:
                    del self.pipeline
                    self.pipeline = None
                
                if self.model:
                    del self.model
                    self.model = None
                
                if self.tokenizer:
                    del self.tokenizer
                    self.tokenizer = None
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                self.is_loaded = False
                self.app_logger.info(f"Model {self.model_name} unloaded successfully")
                
            except Exception as e:
                self.error_logger.error(
                    f"Error during model unload: {str(e)}",
                    exc_info=True
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
        """텍스트 생성을 수행합니다."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        try:
            if self.inference_mode == "pipeline":
                # pipeline 모드에서는 pipeline 객체를 사용
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    **kwargs
                )
                return outputs[0]['generated_text']
            else:
                # tokenize 모드에서는 직접 토큰화 및 생성
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    **kwargs
                )
                
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            self.error_logger.error(
                f"Error during text generation: {str(e)}",
                exc_info=True
            )
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다."""
        return {
            'model_name': self.model_name,
            'model_id': self.model_id,
            'inference_mode': self.inference_mode,
            'device': self.device,
            'is_loaded': self.is_loaded
        }