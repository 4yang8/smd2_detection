import logging
import json
import os
from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime

class Turn(BaseModel):
    role: str
    text: str
# score_mamager 대신 사용 기본은 안 들어가 있는 것. 진단 모드에서는 -1에 대해 3번 묻게 해야 함. 
class ChatTracker:
    def __init__(self):
        # {user id : item}
        self.cesd_info = {} # cesd 점수 관리
        self.chat_mode = {}
        self.cesd_count = {} # 해당 질문 몇 번 물어봤는지
        self.cesd_num = {} # 물어봐야 할 질문 번호
        self.terminate = {}
        self.phase = {}
        self.phase_turn = {}
        self.score = {} # 총점
        self.history_phase = {}
        self.history: Dict[str, List[Turn]] = {} # 사용자별 전체 history
        self.persona: Dict[str, str] = {} # 사용자별 persona
        self.time: Dict[str, str] = {}
        self.asked_labels: Dict[str, set[str]] = {}
        self.turn: Dict[str, int] = {}                   # user 발화 turn, 0부터 시작
        self.persona_user_utts: Dict[str, List[str]] = {}
        
    def _now_str(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def set_time(self, user_id, when:str | None = None):
        self.time[user_id] = when if when else self._now_str()
    
    def get_time(self, user_id) -> str:
        return self.time.get(user_id, "")
    
    def update_persona(self, user_id: str, new_persona: str):
        if not new_persona.strip():
            return  # 아무 내용이 없으면 무시

        self.persona[user_id] = new_persona
    
    def get_persona(self, user_id: str) -> str:
        return self.persona.get(user_id, "")
    
    def get_history(self, user_id: str) -> List[Turn]:
        return self.history.get(user_id, [])
    
    def insert_history(self, user_id: str, text: str, speaker: str, *, append: bool = True) -> None:
        """
        append=False면 기존 history[user_id]를 덮어쓰기 모드로 변경
        """
        turn = Turn(role=speaker, text=text)
        if user_id not in self.history or not append:
            self.history[user_id] = [turn]
        else:
            self.history[user_id].append(turn)
        
        
    def set_cesd_num(self, user_id, num):
        num = int(num)
        self.cesd_num[user_id] = num
    def get_cesd_num(self, user_id):
        if user_id not in self.cesd_num:
            return -1
        return self.cesd_num[user_id]

    def set_cesd_count(self, user_id, num):
        self.cesd_count[user_id] = num
    def get_cesd_count(self, user_id):
        if user_id not in self.cesd_count:
            return 0
        return self.cesd_count[user_id]
    
    def set_phase(self, user_id, phase):
        self.phase[user_id] = phase
    def get_phase(self, user_id):
        if user_id not in self.phase:
            return "exploration"
        return self.phase[user_id]
    
    def set_phase_turn(self, user_id, num):
        self.phase_turn[user_id] = num
    def get_phase_turn(self, user_id):
        if user_id not in self.phase_turn:
            return 1
        return self.phase_turn[user_id]

    def set_score(self, user_id, num):
        self.score[user_id] = num
    def get_score(self, user_id):
        if user_id not in self.score:
            return -1
        return self.score[user_id]
    
    def set_history_phase(self, user_id: str, utt: str, append: bool = True):
        if utt is None:
            return
        text = str(utt)

        if not append:  # ← 복원/로딩: 덮어쓰기
            self.history_phase[user_id] = text
            return

        # 이어붙이기(기존 로직)
        if user_id not in self.history_phase or not self.history_phase[user_id]:
            self.history_phase[user_id] = text
        else:
            self.history_phase[user_id] += f"\n{text}"

    def get_history_phase(self, user_id):
        return self.history_phase.get(user_id, "")
        
    def reset_history_phase(self, user_id):
        self.history_phase[user_id] = ""
        
    
    # PHQ Management
    def get_cesd_info(self, user_id):
        if user_id not in self.cesd_info:
            self.cesd_info[user_id] = {}
        else:
            # 키를 int로 변환 (한 번만 변환하면 됨)
            self.cesd_info[user_id] = {
                int(k): v for k, v in self.cesd_info[user_id].items()
            }
        return self.cesd_info[user_id]
    
    def set_cesd_info(self, user_id: str, info: dict):
        info = info or {}
        self.cesd_info[user_id] = {int(k): v for k, v in info.items()}
    
    def isdone_cesd(self, user_id): 
        if user_id not in self.cesd_info:
            self.cesd_info[user_id] = {}

        return len(self.cesd_info[user_id]) == 20 and all(value != -1 for value in self.cesd_info[user_id].values())


    def get_current_cesd(self, user_id): # 들어있지 않은 or -1인 가장 앞 숫자 num 가져오기
        if user_id not in self.cesd_info:
            return 1

        sorted_keys = sorted(self.cesd_info[user_id].keys())
        for key in range(1, 21):
            if key not in sorted_keys or self.cesd_info[user_id].get(key, -1) == -1:
                return key
        return None
        
    
    def insert_cesd_info(self, user_id, score, item):
        if user_id not in self.cesd_info:
            self.cesd_info[user_id] = {}
        print("여기까지 옴")
        print(item)
        try:
            item = int(item)
        except ValueError:
            print("오류: valueError")
            return
        print(type(item))
        if 1 <= item <= 20:
            print("여기2")
            if item in [4, 8, 12, 16]:
                if score in [0, 1, 2, 3]:
                    score = 3 - score
            self.cesd_info[user_id][item] = score
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)

            # tracker에 기록된 시간 가져오기 (없으면 지금 시간 저장)
            log_time = self.get_time(user_id)

            # user_id 전용 폴더 생성
            user_dir = os.path.join(log_dir, user_id)
            os.makedirs(user_dir, exist_ok=True)

            # 안전한 파일명(공백, 콜론 제거)
            safe_time = log_time.replace(":", "-").replace(" ", "_")
            log_path = os.path.join(user_dir, f"{safe_time}_{user_id}.json")

            # 기존 파일 불러오기 또는 초기화
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {"history": []}

            # ✅ 점수 정렬해서 저장 (키 int 변환)
            normalized = {int(k): v for k, v in self.cesd_info[user_id].items()}
            sorted_items = sorted(normalized.items(), key=lambda kv: kv[0])
            sorted_score = dict(sorted_items)
            data["score"] = sorted_score
            
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
    def has_cesd_item(self, user_id, num):
        if user_id not in self.cesd_info:
            return False  # 유저 정보가 아예 없으면 False

        return num in self.cesd_info[user_id]  # item이 해당 유저의 PHQ-9 정보에 있는지 확인
    
    def get_cesd_result(self, user_id):
        if self.isdone_cesd(user_id):
            sorted_cesd_info = sorted(self.cesd_info[user_id].items(), key = lambda x: x[0])
            scores = [value for _, value in sorted_cesd_info]
            
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)

            # tracker.py 내부이므로 self.get_time 사용
            log_time = self.get_time(user_id)

            # user_id 전용 폴더 생성
            user_dir = os.path.join(log_dir, user_id)
            os.makedirs(user_dir, exist_ok=True)

            # 파일명: {time}_{user_id}.json
            safe_time = log_time.replace(":", "-").replace(" ", "_")
            log_path = os.path.join(user_dir, f"{safe_time}_{user_id}.json")

            # 기존 파일 불러오기 또는 초기화
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {"history": []}

            # 점수 계산
            if -2 in scores or -1 in scores:
                data["total_score"] = "none"
                data["result"] = "진단 불가"
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return "진단 불가", "no score"

            total_score = sum(scores)

            if 0 <= total_score <= 15:
                result = "normal"
            elif 16 <= total_score <= 20:
                result = "mild"
            elif 21 <= total_score <= 24:
                result = "moderate"
            else:
                result = "severe"

            data["total_score"] = total_score
            data["result"] = result

            # 저장
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return total_score, result

    # Mode Management
    def set_chat_mode(self, user_id, mode):
        self.chat_mode[user_id] = mode
        
    
    def get_chat_mode(self, user_id):
        if user_id not in self.chat_mode:
            self.chat_mode[user_id] = 'counseling'
        return self.chat_mode[user_id]
    
    # def get_phq_flag(self, user_id):
    #     if user_id not in self.phq_flag:
    #         self.phq_flag[user_id] = False
    #         return True
    #     return self.phq_flag[user_id]
    
    # Terminate
    def get_terminate(self, user_id):
        if user_id not in self.terminate:
            self.terminate[user_id] = False
        return self.terminate[user_id]

    def set_terminate(self, user_id):
        self.terminate[user_id] = True
        logging.info(f'[TERMINATE] {user_id}')
        
    def set_history(self, user_id: str, history_list: list[dict]) -> None:
        self.history[user_id] = [Turn(role=h["role"], text=h["text"]) for h in history_list]
        
    def get_turn(self, user_id: str) -> int:
        return self.turn.get(user_id, 0)

    def set_turn(self, user_id: str, turn: int):
        self.turn[user_id] = turn

    def increase_turn(self, user_id: str):
        self.turn[user_id] = self.get_turn(user_id) + 1

    def add_persona_utt(self, user_id: str, utt: str):
        if user_id not in self.persona_user_utts:
            self.persona_user_utts[user_id] = []
        if utt is not None and str(utt).strip():
            self.persona_user_utts[user_id].append(str(utt).strip())

    def get_persona_utts(self, user_id: str) -> List[str]:
        return self.persona_user_utts.get(user_id, [])

    def get_persona_input(self, user_id: str) -> str:
        return " ".join(self.get_persona_utts(user_id)).strip()

    def clear_persona_utts(self, user_id: str):
        self.persona_user_utts[user_id] = []
        
    def get_state(self, user_id: str) -> dict:
        return {
        "cesd_info": self.get_cesd_info(user_id),
        "chat_mode": self.get_chat_mode(user_id),
        "cesd_count": self.get_cesd_count(user_id),
        "cesd_num": self.get_cesd_num(user_id),
        "terminate": self.get_terminate(user_id),
        "phase": self.get_phase(user_id),
        "phase_turn": self.get_phase_turn(user_id),
        "score": self.get_score(user_id),
        "history_phase": self.get_history_phase(user_id),
        "persona": self.get_persona(user_id),
        "history": [turn.dict() for turn in self.get_history(user_id)],
        "time": self.get_time(user_id),
        "turn": self.get_turn(user_id),
        "persona_user_utts": self.get_persona_utts(user_id),
    }

    def set_state(self, user_id: str, state: dict):
        self.set_cesd_info(user_id, state.get("cesd_info"))
        self.set_chat_mode(user_id, state.get("chat_mode", "counseling"))
        self.set_cesd_count(user_id, state.get("cesd_count", 0))
        self.set_cesd_num(user_id, state.get("cesd_num", -1))
        if state.get("terminate", False):
            self.set_terminate(user_id)
        self.set_phase(user_id, state.get("phase", "exploration"))
        self.set_phase_turn(user_id, state.get("phase_turn", 1))
        self.set_score(user_id, state.get("score", None))
        self.set_history_phase(user_id, state.get("history_phase", ""), False)
        self.set_turn(user_id, state.get("turn", 0))
        self.persona_user_utts[user_id] = state.get("persona_user_utts", [])
        if state.get("persona"):
            self.update_persona(user_id, state["persona"])
        self.set_history(user_id, state.get("history", []))
        if state.get("time"):  
            self.set_time(user_id, state["time"])
            
    def reset_state(self, user_id: str):
        self.cesd_info[user_id] = {}
        self.chat_mode[user_id] = "counseling"
        self.cesd_count[user_id] = 0
        self.cesd_num[user_id] = -1
        self.terminate[user_id] = False
        self.phase[user_id] = "exploration"
        self.phase_turn[user_id] = 1
        self.score[user_id] = None
        self.history_phase[user_id] = ""
        self.persona[user_id] = {}
        self.history[user_id] = []
        self.time[user_id] = None
        self.turn[user_id] = 0
        self.persona_user_utts[user_id] = []
        print(f"[INFO] State for {user_id} has been reset.")
        
    def get_asked_labels(self, user_id: str) -> set[str]:
        return self.asked_labels.get(user_id, set())


    def add_asked_label(self, user_id: str, label: str):
        if user_id not in self.asked_labels:
            self.asked_labels[user_id] = set()
        self.asked_labels[user_id].add(label)


    def reset_asked_labels(self, user_id: str):
        self.asked_labels[user_id] = set()
        
    def get_missing_labels(self, user_id: str, all_labels: List[str]) -> List[str]:
        covered = self.get_asked_labels(user_id)
        return sorted(set(all_labels) - covered)
