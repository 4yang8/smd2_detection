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

# 24시간 이내 접속 시 이어서 상담 진행, 판단해주는 함수 (접속 시간 및 state)
def ensure_tracker_time_on_connect(tracker, user_id: str, base_dir: str = "logs", now_dt: datetime | None = None) -> None:
    
    now_dt = now_dt or datetime.now()

    print(user_id)
    if not tracker.get_time(user_id):
        now_dt = now_dt or datetime.now()
        user_dir = os.path.join(base_dir, user_id)
        if not os.path.isdir(user_dir):
            tracker.set_time(user_id, now_dt.strftime("%Y-%m-%d %H:%M:%S"))
            return
    user_dir = os.path.join(base_dir, user_id)
    # state_{user_id}_...json 중 가장 최신 파일 찾기
    candidates = [
        f for f in os.listdir(user_dir)
        if f.startswith(f"state_{user_id}_") and f.endswith(".json")
    ]
    if not candidates:
        tracker.set_time(user_id, now_dt.strftime("%Y-%m-%d %H:%M:%S"))
        return
    
    latest_file = sorted(candidates, reverse=True)[0]
    # print(sorted(candidates, reverse=True))
    latest_path = os.path.join(user_dir, latest_file)

    # JSON 내부의 "time" 값 읽기
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

    # ✅ 디버깅 출력
    # print(f"[DEBUG] Now: {now_dt}")
    # print(f"[DEBUG] Last file: {latest_file}")
    # print(f"[DEBUG] Last time in file: {last_time_dt}")

    # 24시간 비교 후 time 설정
    if last_time_dt and (now_dt - last_time_dt) < timedelta(hours=24):
        tracker.set_time(user_id, last_time_dt.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            tracker.set_state(user_id, state)
            # print(f"[INFO] Restored tracker state from {latest_file}")
        except Exception as e:
            print(f"[WARN] Failed to restore tracker state: {e}")
        # print(tracker.get_state(user_id))
        # print(last_time_dt)
    else:
        tracker.reset_state(user_id)
        tracker.set_time(user_id, now_dt.strftime("%Y-%m-%d %H:%M:%S"))
        # print(now_dt)
    
# state를 저장하는 함수
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

    user_dir = os.path.join(base_dir, user_id)
    os.makedirs(user_dir, exist_ok=True)

    # ✅ state_{sender_id}_{safe_time}.json
    state_path = os.path.join(user_dir, f"state_{user_id}_{ts_safe}.json")
    state_data = tracker.get_state(user_id)

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state_data, f, ensure_ascii=False, indent=2)
        

import urllib.parse
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, Request

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
# @app.get("/send_recent_logs", dependencies=[Depends(require_token)])
# async def send_recent_logs(user_id: str):
#     """
#     GET 요청으로 sender_id를 받아 logs/{user_id}/ 폴더 내에서
#     'state_'로 시작하지 않는 가장 최신 JSON 파일 내용을 JSONResponse로 반환.
#     예: http://141.223.163.135:8000/send_json?user_id=default_user
    
#     """
#     try:
#         # ✅ URL 인코딩된 이메일 복원 (예: test%40google.com → test@google.com)
#         user_id = urllib.parse.unquote(user_id)

#         # ✅ 상대 경로: logs/{sender_id}/
#         log_dir = os.path.join("logs", user_id)

#         # 폴더 존재 확인
#         if not os.path.exists(log_dir):
#             return JSONResponse(content={"error": f"Directory not found: {log_dir}"}, status_code=404)

#         # ✅ JSON 파일 목록 (state_로 시작하는 파일 제외)
#         json_files = [
#             os.path.join(log_dir, f)
#             for f in os.listdir(log_dir)
#             if f.endswith(".json") and not f.startswith("state_")
#         ]

#         if not json_files:
#             return JSONResponse(content={"error": f"No valid JSON files found in {log_dir}"}, status_code=404)
        
#         latest_file = max(json_files, key=lambda x: os.path.basename(x))

#         print(f"[SEND_JSON] User: {user_id}")
#         print(f"[SEND_JSON] Latest non-state file: {latest_file}")

#         # ✅ 파일 내용 읽어서 JSONResponse로 반환
#         with open(latest_file, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         return JSONResponse(content=data)

#     except Exception as e:
#         print(f"[SEND_JSON ERROR] {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)
    
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
    if response.is_ending:  # 또는 if done:
        # 1) 먼저 pending 상태 파일 만들어 두고
        background_tasks.add_task(init_summary_file, item.sender_id)
        # 2) 백그라운드에서 GPT 요약 생성 + summary.json 업데이트 return보다 후순위 실행(작업 등록)
        background_tasks.add_task(generate_summary, item.sender_id)
    # print("[DEBUG] RETURNING RESPONSE")
    # print("[RETURN]", item.sender_id, datetime.now().isoformat(timespec="seconds"))
    return response


#=================UNIST CODE=====================
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("UNIST_GPT_API")) 

# summary file setting
def init_summary_file(sender_id: str):        
    summary_path = Path(f"logs/{sender_id}/{sender_id}_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    create_time = datetime.now().isoformat()

    data = {
        "status": "pending",
        "summary": None,
        "created_at": create_time
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_input_data(input_data: dict | list) -> list[dict] | None:
    """입력 데이터를 처리하여 일관된 딕셔너리 리스트 형식으로 반환합니다."""
    if isinstance(input_data, list):
        return input_data
    elif isinstance(input_data, dict) and 'history' in input_data:
        return input_data['history']
    return None

def format_chat_log_to_string(chat_log: list[dict]) -> str:
    """대화 로그를 API에 전달할 단일 문자열로 변환합니다."""
    formatted_string = ""
    for message in chat_log:
        speaker_raw = message.get("speaker")
        if speaker_raw in ["Stage change"]: continue
        speaker = "사용자" if speaker_raw == "User" else "챗봇" if speaker_raw in ["Bot", "Bot response"] else speaker_raw
        text = message.get("text", "")
        if speaker and text:
            formatted_string += f"{speaker}: {text.strip()}\n"
    return formatted_string.strip()

def create_summary_prompt(chat_log_string: str) -> str:
    """상담 대화에 대한 구조화된 JSON 요약 프롬프트를 생성합니다."""
    # (이전 질문에서 제공된 프롬프트 내용과 동일)
    return f"""
        당신은 상담을 마친 경찰관에게 'POLIFE AI의 소견'을 JSON 형식으로 작성하는, 명확하고 지지적인 AI 어시스턴트입니다.
        [분석할 대화 로그]
        {chat_log_string}
        [소견 작성 지시사항]
        AI는 반드시 3개의 key("key1", "key2", "key3")를 가진 JSON 객체 형식으로 결과를 반환해야 합니다.
        1.  "key1" (상황 진단 및 전환): 2~3 문장, 200자 내외. '경찰관님' 호칭 사용. (상황 요약 -> 긍정적 발견 -> 희망적 마무리)
        2.  "key2" (실천 계획): '~하기' 형태의 bullet point(-) 1개.
        3.  "key3" (제안 및 마무리): 활동의 긍정적 효과 언급. 3문장, 200자 내외.
        [결과물 형식 및 제약 조건]
        -   출력은 반드시 JSON 형식이어야 합니다.
        -   결과물에 서식이나 제목(`**`, `\\n` 등)을 절대 포함하지 마세요.
    """

def get_summary_from_gpt(prompt: str) -> str:
    """OpenAI 모델에 프롬프트를 보내고 결과를 받아옵니다."""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI 클라이언트가 설정되지 않았습니다. API 키를 확인해주세요.")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
    
def get_latest_chat_file(sender_id: str) -> Path | None:
    base_dir = Path(f"logs/{sender_id}")
    if not base_dir.exists():
        return None

    # 패턴 예: 2025-10-15_19-41-26_1015police.json
    json_files = sorted(base_dir.glob(f"*_{sender_id}.json"))
    
    if not json_files:
        return None
    
    return json_files[-1]  # 가장 최근 파일

def generate_summary(sender_id: str):
    print("[DEBUG] generate_summary START")

    latest_file = get_latest_chat_file(sender_id)
    created_at = datetime.now().isoformat()

    if latest_file is None:
        print(f"[Summary] No chat file found for sender {sender_id}")
        return

    print(f"[Summary] Using file: {latest_file}")

    # 1) 대화 로그 읽기
    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) history → 문자열 변환
    chat_history = process_input_data(data)
    chat_log_string = format_chat_log_to_string(chat_history)

    # 3) GPT 프롬프트 생성
    prompt = create_summary_prompt(chat_log_string)

    # 4) GPT 요약 생성
    try:
        summary_text = get_summary_from_gpt(prompt)
        status = "done"
    except Exception as e:
        summary_text = f"요약 생성 중 오류: {e}"
        status = "error"

    # 5) summary.json 경로
    summary_path = Path(f"logs/{sender_id}/{sender_id}_summary.json")

    # 기존 created_at 유지
    
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if "created_at" in existing:
                created_at = existing["created_at"]
        except:
            pass

    # 6) summary.json 내용 구성
    summary_data = {
        "status": status,
        "summary": summary_text,
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(),
        "source_file": str(latest_file)
    }

    # 7) summary.json 저장
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print(f"[Summary] Saved summary to {summary_path}")

# summary file 관련 api
#==============UNIST API=========================
# summary file이 생성되었는지 확인하는 api. 프론트에서 먼저 이걸 호출하여 done인지 확인해야 함. (pending/done/error)
@app.get("/summary_status")
async def summary_status(user_id: str):
    summary_path = Path(f"logs/{user_id}/{user_id}_summary.json")

    # 파일이 아직 생성도 안 됨 → 요약 시작 전
    if not summary_path.exists():
        return {"status": "not_started"}

    # summary.json 읽기
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"status": "error"}

    # status 값만 반환 (pending / done / error)
    status = data.get("status", "unknown")
    return {"status": status}

# @app.post("/debug/sleep")
# async def debug_sleep():
#     print("[SLEEP ENTER]", datetime.now().isoformat(timespec="seconds"))
#     await asyncio.sleep(10)
#     print("[SLEEP EXIT ]", datetime.now().isoformat(timespec="seconds"))
#     return {"ok": True}

# done 확인 이후 부르는 api. summary를 보내준다. 
@app.get("/get_summary")
async def get_summary(user_id: str):
    summary_path = Path(f"logs/{user_id}/{user_id}_summary.json")

    # 파일이 없는 경우 → 요약 시작도 안 됨
    if not summary_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "status": "not_started",
                "message": "요약 파일이 없습니다. 대화가 종료되지 않았거나 요약 생성이 시작되지 않았습니다."
            }
        )

    # 파일 읽기
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"요약 파일을 읽는 중 오류 발생: {e}"
            }
        )

    # 상태별 처리
    status = data.get("status")

    if status == "pending":
        return JSONResponse(
            status_code=202,
            content={
                "status": "pending",
                "message": "요약 생성 중입니다. 잠시 후 다시 시도해주세요."
            }
        )

    if status == "error":
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "요약 생성 중 오류가 발생했습니다.",
                "detail": data.get("summary", "")
            }
        )

    if status == "done":
        return JSONResponse(
            status_code=200,
            content={
                "status": "done",
                "summary": data.get("summary"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            }
        )

    # 정의되지 않은 상태 값
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"알 수 없는 summary status: {status}"
        }
    )
