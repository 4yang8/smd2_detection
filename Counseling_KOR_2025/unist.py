from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime
import json, os, re
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.environ.get("UNIST_GPT_API"))

def get_latest_chat_file(user_id: str) -> Path | None:
    base_dir = Path(f"logs/{user_id}")
    if not base_dir.exists():
        return None

    json_files = sorted(base_dir.glob(f"*_{user_id}.json"))
    if not json_files:
        return None

    return json_files[-1]


def format_chat_log_to_string(chat_log):
    formatted = ""
    for message in chat_log:
        speaker_raw = message.get("speaker")
        if speaker_raw == "Stage change":
            continue

        speaker = (
            "사용자" if speaker_raw == "User"
            else "챗봇" if speaker_raw in ["Bot", "Bot response"]
            else speaker_raw
        )

        text = message.get("text", "")
        if speaker and text:
            formatted += f"{speaker}: {text.strip()}\n"

    return formatted.strip()

def extract_score_data(data: dict):
    """
    기존 로그 안의 score를 summary prompt용 score_data로 변환
    """
    score = data.get("score", {})

    if isinstance(score, dict):
        return score

    return {}

def create_summary_prompt(chat_log_string: str, score_data:dict) -> str:
    """상담 대화에 대한 구조화된 JSON 요약 프롬프트를 생성합니다."""
    return f"""
# Role & Persona
당신은 경찰관의 심리적 회복을 돕는 'POLIFE AI' 상담가입니다. 
당신의 말투는 전문적이면서도, 따뜻한 조언을 건네는 인생의 선배나 전문 상담사처럼 다정해야 합니다.
**중요** 모든 상황에서 호칭은 반드시 '경찰관님'을 사용하세요.

# Input Data
1. CES-D Score: {json.dumps(score_data, indent=2)}
2. Chat Log: {chat_log_string}

# Step 1: Internal Analysis (Do not output this part)
1. 점수 계산 (역채점 필수):
   - 요인 1 (대인관계/집중력): 문항 5, 7, 9, 10, 13, 14, 15, 19, 20 평균
   - 요인 2 (신체화 증상): 문항 1, 2, 3, 5, 6, 11 평균
   - 요인 3 (긍정 정서 결핍): 문항 4, 8, 12, 16 (**3 - 점수** 로 역채점 후 평균)
   - 요인 4 (실존적 좌절): 문항 17, 18 평균
2. 블렌딩 전략:
   - 점수가 높은 순으로 Main(1위)과 Sub(2위) 선정.
   - 동점일 경우: 요인 2 > 요인 4 > 요인 1 > 요인 3 순으로 우선순위 부여.
   - 비율: 단독 1위면 Main 90% + Sub 10%, 동점이면 Main 80% + Sub 20% 비율로 분석 내용을 구성할 것.

# Output Specification (JSON Format Only)
반드시 아래 4개의 Key를 가진 JSON으로 응답하세요.

1. "keywords":
   - **제약**: UI 박스 크기를 고려하여 **공백 포함 20자 이내**로 작성할 것.
   - [
       {{ "type": "증상", "content": "현재의 심리적 고충을 비유한 짧은 문구" }},
       {{ "type": "솔루션", "content": "변화와 희망을 담은 짧은 문구" }}
     ]

2. "key1" (상담 요약 - 증상 분석 및 수용):
   - **구체성**: 대화 로그에 등장한 **구체적인 사건이나 단어(예: 커피, 잠, 동료, 민원 등)**를 첫 문장에 반드시 포함하여 대화의 맥락을 즉시 짚어주세요.
   - **문장 고착화 방지**: "요즘 힘든 시기를...", "마음이 무겁게..." 등 뻔한 위로로 시작하지 마세요.
   - **분석 반영**: 위에서 계산한 Main/Sub 요인의 특성을 대화 맥락과 섞어 설명하세요.
   - **어투**: "~거든요", "~해 보여요", "~인 거예요"와 같은 부드러운 구어체를 사용하세요.
   - **분량**: 200자 내외.

3. "key2" (나만의 힐링 도구 - 행동 유도 및 기대효과):
   - **분량**: 250자 내외. 줄글로 설명 (화살표나 괄호 금지).
   - **핵심 지시 (3단계 분기 로직)**: CASE A(약속), B(언급), C(제안) 중 하나를 택하되, '시스템적 용어'나 '약속 여부'를 직접 언급하지 마세요.
     - **CASE A (명확히 약속한 활동이 있는 경우)**: 사용자가 "앞으로 ~해도 좋겠어요" 혹은 "내일부터 ~해볼게요" 등 명시적으로 의사를 밝힌 경우에만 "아까 ~를 하기로 약속하셨잖아요"와 같이 약속을 상기시키며 도입.
     - **CASE B (약속은 없으나 긍정적 행동이 언급된 경우)**: 대화 중 언급된 취미나 긍정적 습관(술, 담배 등 제외)을 포착하여 "말씀하신 ~를 하는 시간이 경찰관님께 큰 힘이 될 것 같아요"라며 권유.
     - **CASE C (행동 언급이 전혀 없는 경우)**: 아무런 단서가 없다면 대화 흐름에 맞춰 아주 간단한 루틴(예: 10분간 생각 멈추기, 스스로에게 칭찬 한마디 하기 등)을 새롭게 제안.
   - **논리(기대효과)**: [해당 활동] -> [신체/뇌의 긍정적 변화] -> [일상 및 업무적 이점]. 해당 행동이 신체/뇌의 긴장을 어떻게 풀고, 일상/업무에 어떤 이점을 주는지 설명. 
   - **금지**: "긍정적인 영향을 줄 거예요", "도움이 될 거예요"와 같은 **추상적인 결론을 금지**합니다.
   - **필수**: "심박수를 안정시켜 교감신경의 흥분을 가라앉히고", "도파민 수용체를 자극하여 무력감을 해소하고" 등 **실질적 이유**를 설명하세요.

4. "key3" (추가 콘텐츠 유도):
   - **역할**: 하단 콘텐츠가 왜 '지금' 이 시점의 경찰관님에게 꼭 필요한지 대화 맥락과 연결하여 설득하세요.
   - **개인화 (필수)**: 대화 로그에서 포착한 경찰관님의 구체적인 고충(예: 밥 먹고 쏟아지는 잠, 사람 간의 서먹함, 풀리지 않는 피로 등)을 문장에 반드시 포함하세요.
   - **문구 구성**: [로그 기반 고민 언급] + [정서적/신체적 효능] + [필수 문구].
   - **필수 포함 표현**: "~해줄 콘텐츠가 아래에 준비되어 있습니다." (이 문구의 형태를 변형하지 말고 자연스럽게 문장에 녹이세요.)
   - **금지**: "준비한 내용을 확인해보세요", "마음을 편안하게 해줄 수 있습니다" 같은 뻔하고 중복되는 표현은 지양하세요.
   - **작성 예시**: 
     - "점심 식사 후마다 찾아오는 **나른한 잠기운을 기분 좋게 깨워줄** 콘텐츠가 아래에 준비되어 있습니다. 이 시간을 통해 오후 일과를 더 가뿐하게 시작해보시길 바라요."

# Negative Constraints
- **절대 금지**: "결과를 가만히 들여다보니", "조금 더 편안해질 방법이 있을까요?" 등 정형화된 문구로 응답을 시작하지 마세요.
- 전문 용어(요인, 척도, 점수) 노출 금지.
- 마크다운 볼드체(**) 금지.
- 반드시 대화체(구어체)로 작성할 것.
"""

def call_gpt(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return json.loads(text)

#==============UNIST API===============

@app.post("/generate_summary")
async def generate_summary(user_id: str):
    latest_file = get_latest_chat_file(user_id)

    if latest_file is None:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "상담 로그 파일을 찾을 수 없습니다."}
        )

    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        chat_history = data.get("history", [])
        score_data = extract_score_data(data)

        chat_log_string = format_chat_log_to_string(chat_history)
        prompt = create_summary_prompt(chat_log_string, score_data)

        summary = call_gpt(prompt)

        now = datetime.now()
        ts = now.strftime("%Y-%m-%d_%H-%M-%S")

        summary_path = Path(f"logs/{user_id}/{ts}_{user_id}_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary_data = {
            "status": "done",
            "summary": summary,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "source_file": str(latest_file)
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        return {
            "status": "done",
            "summary_file": str(summary_path)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/summary_status")
async def summary_status(user_id: str):
    base_dir = Path(f"logs/{user_id}")

    if not base_dir.exists():
        return {"status": "not_started"}

    summary_files = sorted(base_dir.glob(f"*_{user_id}_summary.json"))

    if not summary_files:
        return {"status": "not_started"}

    latest_summary = summary_files[-1]

    try:
        with open(latest_summary, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"status": "error"}

    return {
        "status": data.get("status", "unknown"),
        "summary_file": latest_summary.name
    }


@app.get("/get_summary")
async def get_summary(user_id: str):
    base_dir = Path(f"logs/{user_id}")

    if not base_dir.exists():
        return JSONResponse(
            status_code=404,
            content={"status": "not_started"}
        )

    summary_files = sorted(base_dir.glob(f"*_{user_id}_summary.json"))

    if not summary_files:
        return JSONResponse(
            status_code=404,
            content={"status": "not_started"}
        )

    latest_summary = summary_files[-1]

    with open(latest_summary, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "status": data.get("status"),
        "summary": data.get("summary"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "summary_file": latest_summary.name
    }
