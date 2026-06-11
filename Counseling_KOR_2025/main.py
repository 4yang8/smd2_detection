from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from Counseling_KOR_2025.utils import Chat1_input_demo, Chat1_output_demo
from Counseling_KOR_2025.model import CounselingChat

import json
from datetime import datetime, timedelta  # ⬅️ 시간 기록용
from fastapi import UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
import tempfile, asyncio, aiofiles
import aiofiles.tempfile, io, asyncio, os
from pathlib import Path
import subprocess
from Counseling_KOR_2025.tracker import ChatTracker
import requests
from typing import Optional
import re
import urllib.parse
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, Request

UNIST_API_BASE = "http://141.223.163.135:8001"


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,      
)

chatbot = CounselingChat(use_lora=True)

# 24시간 이내 재접속이면 기존 상담 상태를 복원하고, 아니면 새 상담 상태로 초기화
def ensure_tracker_time_on_connect(tracker, user_id: str, base_dir: str = "logs", now_dt: datetime | None = None) -> None:
    
    now_dt = now_dt or datetime.now()

    print(user_id)
    # tracker에 시간이 없으면 기존 로그 디렉토리 존재 여부 확인
    if not tracker.get_time(user_id):
        now_dt = now_dt or datetime.now()
        user_dir = os.path.join(base_dir, user_id)
        
        # 기존 로그가 없으면 현재 시간을 새 상담 시작 시간으로 설정
        if not os.path.isdir(user_dir):
            tracker.set_time(user_id, now_dt.strftime("%Y-%m-%d %H:%M:%S"))
            return
        
    user_dir = os.path.join(base_dir, user_id)
    # state_{user_id}_...json 중 가장 최신 파일 찾기
    candidates = [
        f for f in os.listdir(user_dir)
        if f.startswith(f"state_{user_id}_") and f.endswith(".json")
    ]
     # 저장된 state 파일이 없으면 현재 시간을 새 상담 시작 시간으로 설정
    if not candidates:
        tracker.set_time(user_id, now_dt.strftime("%Y-%m-%d %H:%M:%S"))
        return
    
    latest_file = sorted(candidates, reverse=True)[0]
    latest_path = os.path.join(user_dir, latest_file)

    # 최신 state 파일에서 마지막 접속 시간 읽기
    last_time_dt = None
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        t_str = state.get("time", "")
        if t_str:
            try:
                last_time_dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    last_time_dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    pass
    except Exception as e:
        print(f"[WARN] Failed to read state time: {e}")

    # 마지막 접속이 24시간 이내이면 기존 state 복원
    if last_time_dt and (now_dt - last_time_dt) < timedelta(hours=24):
        tracker.set_time(user_id, last_time_dt.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            tracker.set_state(user_id, state)
        except Exception as e:
            print(f"[WARN] Failed to restore tracker state: {e}")

    # 24시간이 지났거나 시간이 없으면 새 상담 상태로 초기화
    else:
        tracker.reset_state(user_id)
        tracker.set_time(user_id, now_dt.strftime("%Y-%m-%d %H:%M:%S"))
       
    
# 현재 tracker state를 JSON 파일로 저장
def save_tracker_state(
    tracker,
    user_id: str,
    base_dir: str = "logs",
    safe_time: Optional[str] = None,   # ← 'YYYY-MM-DD HH:MM:SS' 형태 문자열
    dt: Optional[datetime] = None      # ← 없으면 now()로 대체
):
    # safe_time 우선 사용, 없으면 dt(또는 now)로 생성
    if safe_time and isinstance(safe_time, str):
        ts = safe_time  # ex) '2025-08-09 17:05:22'
    else:
        dt = dt or datetime.now()
        ts = dt.strftime("%Y-%m-%d %H:%M:%S")

    ts_safe = ts.replace(" ", "_").replace(":", "-")  # ex) '2025-08-09_17-05-22'
    
    # 사용자 로그 디렉토리 생성
    user_dir = os.path.join(base_dir, user_id)
    os.makedirs(user_dir, exist_ok=True)

    # ✅ state_{sender_id}_{safe_time}.json
    state_path = os.path.join(user_dir, f"state_{user_id}_{ts_safe}.json")
    state_data = tracker.get_state(user_id)

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state_data, f, ensure_ascii=False, indent=2)
    

load_dotenv()
# API TOKEN은 unist와 소통 시 보안을 위한 비밀번호
API_TOKEN = os.getenv("API_TOKEN")  # .env 안에 저장된 토큰을 읽음


# -------------------------------------------------------------
# 1️⃣ 인증 토큰 검사 함수
# -------------------------------------------------------------
async def require_token(request: Request):
    """요청 헤더에서 Bearer 토큰을 추출하고 검증"""
    print("요청")
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )

    token = auth.split(" ", 1)[1]  # 'Bearer ' 뒤의 실제 토큰 부분
    
    if token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid token",
        )
    print("통과!")

# -------------------------------------------------------------
# 2️⃣ 보호된 엔드포인트 (/send_recent_logs)
# -------------------------------------------------------------
# 누군가 이 사람의 가장 최신 상담 기록을 요청할 때 부르라고 하면 되는 api

@app.get("/send_recent_logs", dependencies=[Depends(require_token)])
async def send_recent_logs(user_id: str):
    try:
        user_id = urllib.parse.unquote(user_id)
        log_dir = os.path.join("logs", user_id)

        if not os.path.exists(log_dir):
            return JSONResponse(
                content={"error": f"Directory not found: {log_dir}"},
                status_code=404
            )

        # 날짜_시간_id.json 형식만 허용
        # 예: 2025-12-02_16-26-51_gpttest.json
        pattern = re.compile(
            r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_.+\.json$"
        )

        valid_files = []

        for f in os.listdir(log_dir):
            match = pattern.match(f)

            if not match:
                continue

            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

            valid_files.append((timestamp, os.path.join(log_dir, f)))

        if not valid_files:
            return JSONResponse(
                content={"error": f"No valid timestamped JSON files found in {log_dir}"},
                status_code=404
            )

        latest_file = max(valid_files, key=lambda x: x[0])[1]

        print(f"[SEND_JSON] User: {user_id}")
        print(f"[SEND_JSON] Latest timestamped log file: {latest_file}")

        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return JSONResponse(content=data)

    except Exception as e:
        print(f"[SEND_JSON ERROR] {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# 한 턴 대화를 로그 저장, state 업데이트
def save_turn_and_state_sync(chatbot, item, final_response, safe_time):
    log_dir = os.path.join("logs", item.sender_id)
    os.makedirs(log_dir, exist_ok=True)

    safe_time_for_fn = safe_time.replace(" ", "_").replace(":", "-")
    file_name = f"{safe_time_for_fn}_{item.sender_id}.json"
    log_path = os.path.join(log_dir, file_name)

    now = datetime.now()
    turn_history = [
        {"speaker": "User", "text": item.text, "time": now.isoformat()},
        {"speaker": "Bot", "text": final_response, "time": now.isoformat()},
    ]

    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    if "history" not in data or not isinstance(data["history"], list):
        data["history"] = []
    data["history"].extend(turn_history)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    save_tracker_state(chatbot.tracker, item.sender_id, base_dir="logs", safe_time=safe_time)

@app.get("/send_recent_status")
async def send_recent_status(user_id: str):
    """
    logs/{user_id}/ 폴더 내에서
    'state_'로 시작하는 가장 최신 JSON 파일을 반환
    """
    try:
        # URL 인코딩 복원
        user_id = urllib.parse.unquote(user_id)

        log_dir = os.path.join("logs", user_id)

        # 폴더 존재 확인
        if not os.path.exists(log_dir):
            return JSONResponse(
                content={"error": f"Directory not found: {log_dir}"},
                status_code=404
            )

        # ✅ state_ 파일만 선택
        state_files = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if f.endswith(".json") and f.startswith("state_")
        ]

        if not state_files:
            return JSONResponse(
                content={"error": f"No state JSON files found in {log_dir}"},
                status_code=404
            )

        # ✅ 가장 최신 파일 선택 (현재는 파일명 기준)
        latest_file = max(state_files, key=lambda x: os.path.basename(x))

        print(f"[SEND_STATUS] User: {user_id}")
        print(f"[SEND_STATUS] Latest state file: {latest_file}")

        # 파일 읽기
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return JSONResponse(content=data)

    except Exception as e:
        print(f"[SEND_STATUS ERROR] {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.on_event("startup")
async def startup():
    chatbot.start_batch_worker()

# main 함수. model 파일의 chat을 호출함. 
@app.post("/chat/chat1")
async def chat(item: Chat1_input_demo, background_tasks: BackgroundTasks) -> Chat1_output_demo:
    # print("[ENTER]", item.sender_id, datetime.now().isoformat(timespec="seconds"))
    
    
    await asyncio.to_thread(ensure_tracker_time_on_connect, chatbot.tracker, item.sender_id)
    
    # chat 부르기 전 (cesd 질문 해야 하는지 판단, 이미지 작업, 모드 판단 등 전처리)
    txt, score, counsel_gen, is_ending = await asyncio.to_thread(chatbot.pre_chat, item)
    # print("[AFTER PRE_CHAT]", item.sender_id, datetime.now().isoformat(timespec="seconds"))
    
    # print(txt)
    
    # print("[BEFORE AWAIT BATCH]", item.sender_id, datetime.now().isoformat(timespec="seconds"))
    # print(counsel_gen)
    
    if counsel_gen:
        meta = {
            "txt": txt, # expression
            "score": score, 
            "is_ending": is_ending,
        }
        # chat을 부르는 것. batch에서 자동으로 들어감
        final_response = await chatbot.submit_to_batch(item, meta)
        # print("[AFTER AWAIT BATCH]", item.sender_id, datetime.now().isoformat(timespec="seconds"))
    else:
        final_response = txt

    # ✅ 응답 구성
    # api에 맞게 응답을 구성해줌. (이렇게 보내야 함)
    response = Chat1_output_demo(
        text=[final_response],
        score=str(score).replace('{', '').replace('}', '').replace("'", ''),
        is_ending=is_ending,
    )
    # 시간 가져오기
    safe_time = chatbot.tracker.get_time(item.sender_id)
    background_tasks.add_task(save_turn_and_state_sync, chatbot, item, final_response, safe_time)
    
    # UNIST summary code run
    if response.is_ending:
        background_tasks.add_task(
            requests.post,
            f"{UNIST_API_BASE}/generate_summary",
            params={"user_id": item.sender_id},
            timeout=60
        )

    return response
