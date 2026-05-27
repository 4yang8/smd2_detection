# 이것도 실행해야 함! 거쳐가는 역할
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import UploadFile, File, HTTPException, Query
import subprocess
import aiofiles.tempfile, io, asyncio, os
import json
from fastapi.responses import StreamingResponse
from pathlib import Path
import requests
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,     
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 정적 파일 경로 설정
app.mount("/assets", StaticFiles(directory=os.path.join(BASE_DIR, "assets")), name="assets")

# 기본 라우트: index.html 반환
@app.get("/")
def read_index():
    return FileResponse(os.path.join(BASE_DIR, "html/home", "index.html"))

@app.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File(...)):
    """
    MediaRecorder(webm)·mp3·wav 등을 받아 Whisper-1로 전사.
    반환 형식: { "transcript": "..." }
    """
    print("speech-to-text.app.py")
 
    # 0) 파일 포맷 체크 (선택)  -----------------------------------
    if file.content_type not in {
        "audio/webm", "audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg"
    }:
        raise HTTPException(status_code=415, detail="Unsupported audio format")

    # 1) 원본 임시 파일 저장
    input_suffix = Path(file.filename).suffix or ".webm"
    async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp_in:
        await tmp_in.write(await file.read())
        input_path = tmp_in.name

    # 2) 변환된 WAV 파일 경로 지정
    output_path = input_path.replace(input_suffix, ".wav")

    # 3) ffmpeg로 WAV로 변환
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
            check=True,
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Failed to convert audio to WAV format")

    # Whisper 호출 (비동기 클라이언트 아님, HTTP proxy 방식) ------------------------
    url = "http://platon.postech.ac.kr:14000/asr/asr"
    headers = {}
    data = {'language': 'Korean'}

    try:
        with open(output_path, "rb") as f:
            files = {
                'file': ('[PROXY]', f, 'audio/wav'),  # webm → wav 로 변경
            }
            response = json.loads(requests.post(url, data=data, files=files).text)
            
            transcription = response['transcription'].strip() if response else ""
            # transcription = response.transcription
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper 요청 실패: {e}")
    finally:
        Path(output_path).unlink(missing_ok=True)  # 임시 파일 삭제
        Path(input_path).unlink(missing_ok=True)  # 임시 파일 삭제

    return {"transcript": transcription.strip()}

@app.get("/tts")
async def tts(text: str = Query(..., max_length=500)):

    headers = {
    "Content-Type": "application/json"
    }
    print("tts.app.py")
    payload = {
        "text": text,
        "speaker": "0"
    }
    url = "http://platon.postech.ac.kr:14000/tts/tts"


    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(response)

    # 응답은 bytes
    audio_bytes = response.content  
    audio_io = io.BytesIO(audio_bytes)
    audio_io.seek(0)

    return StreamingResponse(
        audio_io,
        media_type="audio/mpeg",
        headers={"Cache-Control": "public, max-age=86400"}   # 간단 캐싱
    )

BACKEND_URL = "https://localhost:8000/chat/chat1"

from fastapi import Request
from fastapi.responses import Response
from fastapi.responses import JSONResponse

@app.options("/chat/chat1")
async def options_chat1():
    return Response(
        content="ok",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )
    
import httpx

@app.on_event("startup")
async def startup():
    app.state.client = httpx.AsyncClient(verify=False, timeout=300.0)

@app.on_event("shutdown")
async def shutdown():
    await app.state.client.aclose()
    
    
@app.post("/chat/chat1")
async def chat(request: Request):
    payload = await request.json()
    print("app.py의 chat1")

    r = await app.state.client.post(BACKEND_URL, json=payload)
    r.raise_for_status()
    data = r.json()
    print(data)

    return JSONResponse(
        content=data,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(">>> INCOMING:", request.method, request.url.path)
    response = await call_next(request)
    return response


#========UNIST API=============

BACKEND_URL_SUMMARY_STATUS = "https://localhost:8000/summary_status"
BACKEND_URL_GET_SUMMARY = "https://localhost:8000/get_summary"

@app.get("/summary_status")
async def summary_status(user_id: str):
    print("app.py의 summary_status")
    backend_url = f"{BACKEND_URL_SUMMARY_STATUS}?user_id={user_id}"
    response = requests.get(backend_url, verify=False)
    return response.json()


@app.get("/get_summary")
async def get_summary(user_id: str):
    print("app.py의 get_summary")
    backend_url = f"{BACKEND_URL_GET_SUMMARY}?user_id={user_id}"
    response = requests.get(backend_url, verify=False)
    return response.json()
