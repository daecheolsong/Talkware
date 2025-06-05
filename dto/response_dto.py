"""
Response DTOs for Talkware API
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Usage(BaseModel):
    """토큰 사용량"""
    prompt_tokens: int = Field(..., description="프롬프트 토큰 수")
    completion_tokens: int = Field(..., description="생성된 토큰 수")
    total_tokens: int = Field(..., description="전체 토큰 수")


class AssistantMessage(BaseModel):
    """어시스턴트 메시지"""
    role: str = Field("assistant", description="메시지 역할")
    content: str = Field(..., description="메시지 내용")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="도구 호출 목록")


class Choice(BaseModel):
    """생성 선택"""
    index: int = Field(..., description="선택 인덱스")
    message: AssistantMessage = Field(..., description="어시스턴트 메시지")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="로그 확률")
    finish_reason: Optional[str] = Field(None, description="생성 종료 이유")
    seed: Optional[int] = Field(None, description="생성 시드")


class LLMResponseDTO(BaseModel):
    """LLM 응답 DTO"""
    id: str = Field(..., description="응답 ID")
    object: str = Field("chat.completion", description="응답 객체 타입")
    created: int = Field(..., description="생성 시간 (Unix timestamp)")
    model: str = Field(..., description="모델 이름")
    prompt: List[Dict[str, Any]] = Field(default_factory=list, description="프롬프트")
    choices: List[Choice] = Field(..., description="생성 선택 목록")
    usage: Usage = Field(..., description="토큰 사용량")
    inference_time: Optional[float] = Field(None, description="추론 시간 (초)") 