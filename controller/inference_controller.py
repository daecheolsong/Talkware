"""
추론 요청을 처리하는 컨트롤러
"""

import json
import time
import uuid
from flask import Blueprint, request, jsonify
from typing import Dict, Any, Optional, List, Union
import logging
from service.model.base_model_loader import BaseModelLoader
from dto.request_dto import InferenceRequestDTO, Message, ContentItem
from dto.response_dto import LLMResponseDTO, Choice, AssistantMessage, Usage


# Blueprint 생성
inference_bp = Blueprint('inference', __name__)

# 모델 인스턴스 저장
_model_instance = None
_loggers = None


def init_controller(model: BaseModelLoader, loggers: Dict[str, logging.Logger]):
    """컨트롤러 초기화
    
    Args:
        model: 로드된 모델 인스턴스 (BaseModelLoader를 상속한 모든 모델 지원)
        loggers: 로거 딕셔너리
    """
    global _model_instance, _loggers
    _model_instance = model
    _loggers = loggers


def _extract_prompt_texts(messages: List[Message]) -> List[str]:
    """메시지 목록에서 텍스트 내용만 추출
    
    Args:
        messages: Message 객체 리스트
        
    Returns:
        List[str]: 추출된 텍스트 목록
    """
    texts = []
    
    for message in messages:
        if isinstance(message.content, str):
            texts.append(message.content)
        else:
            # ContentItem 리스트인 경우
            for item in message.content:
                if isinstance(item, ContentItem):
                    texts.append(item.text)
                elif isinstance(item, dict) and item.get('type') == 'text':
                    texts.append(item['text'])
    
    return texts


def _extract_prompt(messages: List[Message]) -> str:
    """메시지 목록에서 프롬프트 추출
    
    Args:
        messages: Message 객체 리스트
        
    Returns:
        str: 추출된 프롬프트 문자열
    """
    return '\n'.join(_extract_prompt_texts(messages))


@inference_bp.route('/generate', methods=['POST'])
async def generate():
    """텍스트 생성 엔드포인트
    
    Request Body:
        {
            "model": "모델명",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "시스템 메시지"}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "사용자 메시지"}]
                }
            ],
            "temperature": 0.7,        # 선택적, 기본값 0.7
            "top_k": 50,              # 선택적, 기본값 50
            "top_p": 0.9,             # 선택적, 기본값 0.9
            "repetition_penalty": 1.2, # 선택적, 기본값 1.2
            "return_full_text": false, # 선택적, 기본값 false
            "truncate": 4096,         # 선택적, 기본값 4096
            "do_sample": true,        # 선택적, 기본값 true
            "num_beams": 1,           # 선택적, 기본값 1
            "num_return_sequences": 1, # 선택적, 기본값 1
            "max_output_tokens": 4096  # 선택적, 기본값 4096
        }
    
    Returns:
        {
            "id": "chat-completion-id",
            "object": "chat.completion",
            "created": timestamp,
            "model": "모델명",
            "prompt": [...],
            "choices": [...],
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 456,
                "total_tokens": 579
            },
            "inference_time": 1.234  # 추론 소요 시간 (초)
        }
    """
    try:
        # 요청 데이터 파싱 및 검증
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Missing request body'
            }), 400
            
        try:
            # DTO로 변환 및 검증
            request_dto = InferenceRequestDTO(**data)
        except ValueError as e:
            return jsonify({
                'error': str(e)
            }), 400
            
        # 프롬프트 텍스트 추출
        prompt_texts = _extract_prompt_texts(request_dto.messages)
        prompt = '\n'.join(prompt_texts)
            
        # 로깅 시작
        request_start_time = time.perf_counter()  # 더 정확한 시간 측정을 위해 perf_counter 사용
        _loggers['access'].info(
            "Inference request received",
            extra={
                'prompt_length': len(prompt),
                'remote_addr': request.remote_addr,
                'model': request_dto.model,
                'generation_params': {
                    'temperature': request_dto.temperature,
                    'top_k': request_dto.top_k,
                    'top_p': request_dto.top_p,
                    'repetition_penalty': request_dto.repetition_penalty,
                    'return_full_text': request_dto.return_full_text,
                    'truncate': request_dto.truncate,
                    'do_sample': request_dto.do_sample,
                    'num_beams': request_dto.num_beams,
                    'num_return_sequences': request_dto.num_return_sequences,
                    'max_output_tokens': request_dto.max_output_tokens
                }
            }
        )

        _loggers['inference'].info("Inference request received \n %s",request_dto)


        
        # 모델 추론 실행
        inference_start_time = time.perf_counter()
        
        # 모든 생성 파라미터를 딕셔너리로 구성
        generation_kwargs = {
            'temperature': request_dto.temperature,
            'top_k': request_dto.top_k,
            'top_p': request_dto.top_p,
            'repetition_penalty': request_dto.repetition_penalty,
            'return_full_text': request_dto.return_full_text,
            'truncate': request_dto.truncate,
            'do_sample': request_dto.do_sample,
            'num_beams': request_dto.num_beams,
            'num_return_sequences': request_dto.num_return_sequences,
            'max_output_tokens': request_dto.max_output_tokens
        }
        
        # None이 아닌 값만 필터링
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
            
        generated_text = await _model_instance.generate(
            prompt=prompt,
            **generation_kwargs
        )
        
        # 추론 시간 계산 (초 단위, 소수점 3자리까지)
        inference_end_time = time.perf_counter()
        inference_time = round(inference_end_time - inference_start_time, 3)
        total_time = round(inference_end_time - request_start_time, 3)
        
        # 토큰 수 계산 (간단한 방식)
        prompt_tokens = len(prompt.split())
        completion_tokens = len(generated_text.split())
        total_tokens = prompt_tokens + completion_tokens
        
        # LLaMA API 형식으로 응답 생성
        response = LLMResponseDTO(
            id=f"nwSe32Y-{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[:12]}",
            object="chat.completion",
            created=int(time.time()),
            model=getattr(_model_instance, 'MODEL_ID', request_dto.model),
            prompt=[{"role": msg.role, "content": msg.content} for msg in request_dto.messages],
            choices=[
                Choice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=generated_text,
                        tool_calls=[]
                    ),
                    finish_reason="stop",
                    logprobs=None,
                    seed=None
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            ),
            inference_time=inference_time  # 추론 시간 (초)
        )
        
        # 로깅에 추론 시간과 생성 파라미터 추가
        _loggers['access'].info(
            "Text generation completed",
            extra={
                'model': request_dto.model,
                'inference_time': inference_time,  # 순수 추론 시간
                'total_time': total_time,         # 전체 요청 처리 시간
                'total_tokens': total_tokens,
                'generation_params': generation_kwargs,
                'tokens_per_second': round(total_tokens / inference_time, 2) if inference_time > 0 else 0
            }
        )


        _loggers['inference'].info("Inference response \n %s", generated_text)
            
        
        
        return jsonify(response.model_dump())
        
    except Exception as e:
        # 에러 로깅
        _loggers['error'].error(
            "Inference request failed",
            exc_info=True,
            extra={
                'error': str(e),
                'remote_addr': request.remote_addr,
                'request_data': data if 'data' in locals() else None,
                'elapsed_time': round(time.perf_counter() - request_start_time, 3) if 'request_start_time' in locals() else None
            }
        )
        # 에러 응답
        return jsonify({
            'error': str(e)
        }), 500


@inference_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """모델 정보 조회 엔드포인트
    
    Returns:
        {
            "model_name": "모델 이름",
            "device": "cuda/cpu",
            "is_loaded": true/false,
            "gpu_memory_allocated_mb": 1234.56,  # GPU 사용 시
            "gpu_memory_reserved_mb": 2345.67    # GPU 사용 시
        }
    """
    try:
        model_info = _model_instance.get_model_info()
        return jsonify({
            'model_name': _model_instance.model_name,
            'device': model_info['device'],
            'is_loaded': _model_instance.is_loaded,
            'gpu_memory_allocated_mb': model_info.get('gpu_memory_allocated_mb', 0),
            'gpu_memory_reserved_mb': model_info.get('gpu_memory_reserved_mb', 0)
        })
    except Exception as e:
        _loggers['error'].error(
            "Failed to get model info",
            exc_info=True,
            extra={'error': str(e)}
        )
        return jsonify({
            'error': str(e),
            'type': e.__class__.__name__
        }), 500 