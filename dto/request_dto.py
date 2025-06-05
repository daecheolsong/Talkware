"""
Request DTOs for Talkware API
"""

from typing import List, Union, Optional, Dict, Any
from pydantic import BaseModel, Field


class ContentItem(BaseModel):
    """콘텐츠 아이템"""
    type: str = Field(..., description="콘텐츠 타입 (예: text)")
    text: str = Field(..., description="콘텐츠 텍스트")


class Message(BaseModel):
    """메시지"""
    role: str = Field(..., description="메시지 역할 (system, user, assistant)")
    content: Union[str, List[ContentItem]] = Field(..., description="메시지 내용")


class InferenceRequestDTO(BaseModel):
    """추론 요청 DTO"""
    model: str = Field(..., description="모델 이름")
    messages: List[Message] = Field(..., description="메시지 목록")
    temperature: Optional[float] = Field(0.7, description="생성 온도")
    top_k: Optional[int] = Field(50, description="상위 k개 토큰만 고려")
    top_p: Optional[float] = Field(0.9, description="누적 확률 임계값")
    repetition_penalty: Optional[float] = Field(1.2, description="반복 패널티")
    return_full_text: Optional[bool] = Field(False, description="전체 텍스트 반환 여부")
    truncate: Optional[int] = Field(4096, description="입력 텍스트 최대 길이")
    do_sample: Optional[bool] = Field(True, description="샘플링 사용 여부")
    num_beams: Optional[int] = Field(1, description="빔 서치 개수")
    num_return_sequences: Optional[int] = Field(1, description="반환할 시퀀스 수")
    max_output_tokens: Optional[int] = Field(4096, description="최대 출력 토큰 수")
    stop_sequences: Optional[List[str]] = Field(None, description="생성 중단 시퀀스 목록")
    stream: Optional[bool] = Field(False, description="스트리밍 여부")
    request_id: Optional[str] = Field(None, description="요청 ID") 