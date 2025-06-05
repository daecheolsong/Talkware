# 모델 로더 초기화 및 로드
model_loader = ModelLoaderClass(
    model_name=model_name,
    config_loader=model_config_loader,
    loggers=loggers,
    inference_mode=inference_mode or "pipeline"  # 기본값은 pipeline 모드
) 