#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────────
# talkware 서비스 실행을 위한 환경 설정 및 제어 스크립트
# 이 스크립트는 Flask 기반의 talkware 애플리케이션을
# - 초기화(init)
# - 시작(start)
# - 중지(stop)
# - 재시작(restart)
# - 상태(status)
# - PID 출력(pid)
# 기능을 제공합니다.
# ────────────────────────────────────────────────────────────────────────────────

# -------------------------------------------------------------------------------
# 1) Flask 애플리케이션 환경 변수 설정
# -------------------------------------------------------------------------------
export FLASK_APP=talkware        # Flask가 load할 애플리케이션 모듈 이름
export FLASK_DEBUG=true          # 자동 reload + 디버깅 모드 활성화

# -------------------------------------------------------------------------------
# 2) 서버 기본 설정 (외부에서 값이 주어지지 않았을 경우 디폴트 할당)
# -------------------------------------------------------------------------------
# 외부 환경변수 SERVER_PORT, SERVER_HOST, SERVER_HOME 이 설정되어 있으면
# 그 값을 사용하고, 아니면 아래 기본값을 사용
SERVER_PORT=${SERVER_PORT:-4003}                   # 서버 포트 (기본 4000)
SERVER_HOST=${SERVER_HOST:-0.0.0.0}                # 바인딩 호스트 (기본 모든 인터페이스)
SERVER_HOME=${SERVER_HOME:-/data/chatbot/sdc3031/talkware} # 애플리케이션 베이스 디렉토리
SERVER_LOG_FILE=${SERVER_HOME}/logs/talkware.log   # 로그 파일 경로

# -------------------------------------------------------------------------------
# 3) 내부 디렉토리 및 파일 경로 설정
# -------------------------------------------------------------------------------
SERVER_BIN=${SERVER_HOME}/bin                 # 실행 스크립트 위치
SERVER_CONFIG_DIR=${SERVER_HOME}/config       # 설정 파일 디렉토리
SERVER_START_FILE=starter.py                  # 애플리케이션 진입점 스크립트
SERVER_PID_FILE=${SERVER_HOME}/run/server.pid # 프로세스 ID 저장 파일

# -------------------------------------------------------------------------------
# 4) Conda 가상환경 설정
# -------------------------------------------------------------------------------
CONDA_ENV_NAME=/data/chatbot/sdc3031/myenv             # 프로젝트별 가상환경 디렉토리
CONDA_SH=~/miniforge3/etc/profile.d/conda.sh   # conda 초기화 스크립트 경로

# -------------------------------------------------------------------------------
# 5) 컨트롤 스크립트용 환경 변수 export
# -------------------------------------------------------------------------------
# talkwarectl.sh 등 외부 스크립트 또는 서비스 매니저가 참조할 변수
export TALKWARE_SERVER_PORT=${SERVER_PORT}
export TALKWARE_SERVER_HOST=${SERVER_HOST}
export TALKWARE_SERVER_BASEDIR=${SERVER_HOME}

# ────────────────────────────────────────────────────────────────────────────────
#   모델 디렉토리 경로 (외부 설정이 없으면 기본값 사용)
# ────────────────────────────────────────────────────────────────────────────────
export TALKWARE_MODEL_BASEDIR=${TALKWARE_MODEL_BASEDIR:-/data/models}
export TALKWARE_HF_HUB_CACHE=${TALKWARE_HF_HUB_CACHE:-/data/huggingface/hub}

# 필요할 경우 conda 환경을 활성화하는 예시
# source $CONDA_SH
# conda activate $CONDA_ENV_NAME

# -------------------------------------------------------------------------------
# 6) 공통 유틸 함수 정의
# -------------------------------------------------------------------------------

# 오류 발생 시 메시지 출력 후 즉시 종료
function error_exit() {
    echo "Error: $1" >&2   # 표준 에러로 메시지 출력
    exit 1                 # 비정상 종료 코드
}

# Conda 가상환경으로 진입
function go_conda_venv() {
    # conda 초기화 스크립트가 없으면 에러
    if [ ! -f "${CONDA_SH}" ]; then
        error_exit "Conda initialization script not found at ${CONDA_SH}"
    fi
    # 스크립트 소싱 및 환경 활성화
    source "${CONDA_SH}" || error_exit "Failed to source conda.sh"
    conda activate "${CONDA_ENV_NAME}" || error_exit "Failed to activate conda environment"
}

# 애플리케이션 실행에 필요한 디렉토리(run, logs) 생성
function init() {
    local directories=(
        "${SERVER_HOME}/run"
        "${SERVER_HOME}/logs"
    )
    for dir in "${directories[@]}"; do
        if [[ ! -d "${dir}" ]]; then
            mkdir -p "${dir}" || error_exit "Failed to create directory: ${dir}"
        fi
    done
}

# -------------------------------------------------------------------------------
# 7) 서버 제어 함수
# -------------------------------------------------------------------------------

function start() {
    go_conda_venv || error_exit "❌ Conda activate 실패"

    export CUDA_VISIBLE_DEVICES=3
    
    echo "CUDA ORDINAL : ${CUDA_VISIBLE_DEVICES}"
    
    if is_running; then
        echo "⚠️ Server already running on port ${SERVER_PORT}"
        return 1
    fi

    echo "🚀 Starting the CAS Gen AI Daemon on port ${SERVER_PORT}..."

    local log_file="${SERVER_HOME}/logs/server.log"
    local err_log_file="${SERVER_HOME}/logs/server_error.log"

    mkdir -p "$(dirname "$log_file")"
    : > "$log_file"
    : > "$err_log_file"

    nohup python "${SERVER_HOME}/${SERVER_START_FILE}" \
        > "$log_file" 2> "$err_log_file" &

    local pid=$!
    echo "$pid" > "$SERVER_PID_FILE"

    # 실행 대기 후 에러 여부 검사
    sleep 10

    if is_running; then
        # 로그에 에러가 있는지 확인
        if grep -i "Traceback" "$err_log_file" > /dev/null; then
            echo "❌ Server crashed immediately. Check error log:"
            tail -n 20 "$err_log_file"
            kill -9 "$pid" 2>/dev/null
            rm -f "$SERVER_PID_FILE"
            return 1
        else
            echo "✅ Server started successfully (PID: $pid)"
            echo "🌍 Server available at: http://$(hostname -I | awk '{print $1}'):${SERVER_PORT}"
            return 0
        fi
    else
        echo "❌ Server process exited. Last error log:"
        tail -n 20 "$err_log_file"
        rm -f "$SERVER_PID_FILE"
        return 1
    fi
}

# 서버 중지
function stop() {
    go_conda_venv   # Conda 환경 진입

    # PID 파일이 없으면 이미 중지된 상태
    if [ ! -f "${SERVER_PID_FILE}" ]; then
        echo "Server is not running (PID file not found)"
        return 0
    fi

    local pid
    pid=$(cat "${SERVER_PID_FILE}")
    # PID 파일이 비어있거나 잘못되었으면 삭제 후 종료
    if [ -z "${pid}" ]; then
        echo "Invalid PID file"
        rm "${SERVER_PID_FILE}"
        return 1
    fi

    # 프로세스가 살아있으면 정상 종료 시도
    if kill -0 "${pid}" 2>/dev/null; then
        echo "Stopping server (PID: ${pid})..."
        kill "${pid}"

        # 최대 10초간 정상 종료 대기
        for i in {1..10}; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                rm "${SERVER_PID_FILE}"
                echo "Server stopped successfully"
                return 0
            fi
            sleep 1
        done

        # 정상 종료되지 않으면 강제 종료
        echo "Force stopping server..."
        kill -9 "${pid}" 2>/dev/null
        rm "${SERVER_PID_FILE}"
        echo "Server force stopped"
    else
        echo "Server is not running (PID: ${pid})"
        rm "${SERVER_PID_FILE}"
    fi
}

# 서버 상태 확인
function status() {
    go_conda_venv   # Conda 환경 진입

    if is_running; then
        local pid
        pid=$(cat "${SERVER_PID_FILE}")
        echo "Server is running (PID: ${pid})"
        ps -f -p "${pid}"    # 프로세스 상세 정보 출력
    else
        echo "Server is not running"
    fi
}

# 실행 중인지 확인 (PID 파일 + kill -0 체크)
function is_running() {
    if [ -f "${SERVER_PID_FILE}" ]; then
        local pid
        pid=$(cat "${SERVER_PID_FILE}")
        if kill -0 "${pid}" 2>/dev/null; then
            return 0    # 실행 중
        fi
    fi
    return 1            # 실행 중 아님
}

# PID 출력
function get_pid() {
    if [ -f "${SERVER_PID_FILE}" ]; then
        cat "${SERVER_PID_FILE}"
    else
        echo "PID file not found"
        return 1
    fi
}

# -------------------------------------------------------------------------------
# 8) 메인 로직: 커맨드 인자에 따라 함수 호출
# -------------------------------------------------------------------------------
case "$1" in
    start)
        init
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 2
        start
        ;;
    status)
        status
        ;;
    pid)
        get_pid
        ;;
    *)
        # 잘못된 인자가 들어온 경우 사용법 안내 및 종료
        echo "Usage: $0 {start|stop|restart|status|pid}"
        exit 1
        ;;
esac

exit 0
