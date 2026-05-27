from dotenv import load_dotenv
load_dotenv()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING" 
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModelForCausalLM
from Counseling_KOR_2025.utils import Turn
from Counseling_KOR_2025.tracker import ChatTracker
import yaml
import logging
import json
import re
from Counseling_KOR_2025.persona import PersonaExtractor
import io
import base64
from io import BytesIO
from openai import OpenAI, AsyncOpenAI
from PIL import Image
import numpy as np
from deepface import DeepFace
from datetime import datetime, timedelta
# import deepspeed
from dotenv import load_dotenv
from vllm import SamplingParams, LLM
from vllm.lora.request import LoRARequest
from collections import defaultdict

import asyncio
from dataclasses import dataclass
import httpx
from Counseling_KOR_2025.missing_labels import get_all_labels
from Counseling_KOR_2025.missing_labels import classify_label
client = AsyncOpenAI(api_key=os.environ.get("POSTECH_GPT_API"))


@dataclass
class BatchItem:
    item: any
    meta: dict
    future: asyncio.Future 

class CounselingChat:
    # 모델 불러오기 prompt 불러오기 등
    def __init__(self, 
                  base_model_name: str = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
                  llama_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                  use_lora: bool = False
                  ):
            
        self.llama_base_model = AutoModelForCausalLM.from_pretrained(
            llama_model_name, 
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float16,
            device_map={"": "cuda:1"}
        )
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            llama_model_name,
            trust_remote_code = True
        )
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        
        symptom_adapter_path = "Counseling_KOR_2025/models/CESD_detector_ver2"
        freq_adapter_path = "Counseling_KOR_2025/models/cesd_freq1103"
        self.llama_model = PeftModelForCausalLM.from_pretrained(
            self.llama_base_model,
            symptom_adapter_path,
            adapter_name="symptom",
            device_map={"": "cuda:1"}
        )
        self.llama_model.load_adapter(freq_adapter_path, adapter_name="freq")
        self.llama_model.eval()
        
        self.freq_tokenizer = AutoTokenizer.from_pretrained(freq_adapter_path)
        self.symptom_tokenizer = AutoTokenizer.from_pretrained(symptom_adapter_path)
        
        self.llama_base_model.eval()
        self.llama_model.eval()
        self.personaExtractor = PersonaExtractor()
        
        self.label_examples = self._load_label_examples()
        
        self.batch_queue = asyncio.Queue()
        self.batch_task = None
        
        self.tracker = ChatTracker()
        # vllm (chatbot)
        self.vllm = httpx.AsyncClient(
            base_url = "http://141.223.163.135:9000/v1",
            timeout = 60.0
        )
        with open('Counseling_KOR_2025/data/counsel_prompt.yaml') as file:
            self.prompts = yaml.load(file, Loader=yaml.FullLoader)

        
    def get_emotion(self, item):
        # 얼굴 표정 있을 때 사용
        # 이미지 -> 감정 변환 함수.
        try:
            # Base64로 인코딩된 이미지 분리
            if not hasattr(item, 'image') or not item.image:
                raise ValueError("이미지 데이터가 없습니다.")
            
            header, encoded = item.image.split(",", 1)

            # Base64 디코딩
            image_data = base64.b64decode(encoded)

            # PIL 이미지로 변환 후 저장
            img_out = Image.open(io.BytesIO(image_data))
            output_path = '/home/yeajinmin/Counseling_KOR_2025/pic/output.png'
            img_out.save(output_path)

            # DeepFace 분석
            results = DeepFace.analyze(img_path=output_path, actions=['emotion'], enforce_detection=False)

            # 분석된 감정 추출
            emotion = results[0]['dominant_emotion']
            # print("emotion: " + emotion)

        except Exception as e:
            # print(f"감정 분석 중 오류 발생: {e}")
            emotion = 'neutral'
        
        finally:
            # 임시 파일 삭제
            output_path = '/home/yeajinmin/Counseling_KOR_2025/pic/output.png'
            if os.path.exists(output_path):
                os.remove(output_path)

        return emotion
    
    
    def start_batch_worker(self):
        # 배치 처리 워커가 실행 중이 아니면 새로 시작
        if self.batch_task is None or self.batch_task.done():
            self.batch_task = asyncio.create_task(self._batch_worker())
        
    async def submit_to_batch(self, item, meta):
        # 요청을 배치 큐에 넣고 결과가 생성될 때까지 대기
        loop = asyncio.get_running_loop()      
        fut = loop.create_future()
        await self.batch_queue.put(BatchItem(item = item, meta = meta, future = fut))
        return await fut
    
    async def _batch_worker(self):
        # 큐에 쌓인 요청들을 일정 크기/시간 기준으로 묶어 batch generation 수행
        MAX_BATCH_SIZE = 2
        MAX_WAIT_TIME = 0.1
        
        while True:
            first = await self.batch_queue.get()
            batch = [first]

            # batch size 또는 wait time 기준으로 요청 추가 수집
            deadline = asyncio.get_running_loop().time() + MAX_WAIT_TIME
            while True:
                if len(batch) >= MAX_BATCH_SIZE:
                    break
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    batch.append(
                        await asyncio.wait_for(self.batch_queue.get(), timeout)
                    )
                except asyncio.TimeoutError:
                    break

            # result 모드가 아닌 것만 phase 계산
            non_result_reqs = []
            for req in batch:
                mode = self.tracker.get_chat_mode(req.item.sender_id)
                if mode != "result":
                    non_result_reqs.append(req)

            if non_result_reqs:
                await asyncio.gather(*[self.get_phase(req.item) for req in non_result_reqs])

            # batch 단위 generation 수행
            gen_results = await asyncio.gather(
                *[self.generate(self.tracker.get_chat_mode(req.item.sender_id), req.item, req.meta) for req in batch]
            )

            # 생성 결과 저장 및 요청 future 반환
            for req, out in zip(batch, gen_results):
                sender_id = req.item.sender_id
                mode = self.tracker.get_chat_mode(sender_id)

                final_text = out
                self.tracker.insert_history(sender_id, final_text, "Counselor")
                self.tracker.set_history_phase(sender_id, "Counselor: " + final_text)

                if mode == "result":
                    self.tracker.set_terminate(sender_id)

                req.future.set_result(out)


    async def phase_check(self, item) -> bool: # 다음 phase로 넘어가도 되는지 판단
        sender_id = item.sender_id
        phase = self.tracker.get_phase(sender_id)
        history = self.tracker.get_history_phase(sender_id)
        persona = self.tracker.get_persona(sender_id)
        
        if phase == "exploration":
            yn_prompt = self.prompts["exploration_check"].format(history = history, persona = persona)
        elif phase == "resolution":
            yn_prompt = self.prompts["resolution_check"].format(history = history, persona = persona)
        else:
            yn_prompt = self.prompts["closing_check"].format(history = history, persona = persona)
            
        messages = [
            {
                "role": "system",
                "content": yn_prompt
            },
            {
                "role": "user",
                "content": history
            },
        ]
        
        payload = {
            "model": "counseling",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 64
        }
        
        r = await self.vllm.post("/chat/completions", json=payload)
        r.raise_for_status()
        
        text = r.json()["choices"][0]["message"]["content"].strip()
        
        m = re.search(r"Final:\s*(YES|NO)", text)
        return bool(m and m.group(1) == "YES")
    
    
    async def get_phase(self, item):
         # 현재 phase의 턴 수와 전환 조건을 확인하고, 필요하면 다음 phase로 변경
        phase_transition = {
            "exploration": {"next": "resolution", "min_turns": 5, "max_turns": 5},
            "resolution": {"next": "closing", "min_turns": 3, "max_turns": 5},
            "closing": {"next": "cesd_start", "min_turns": 1, "max_turns": 1}
        }

        sender_id = item.sender_id
        current_phase = self.tracker.get_phase(sender_id)
        info = phase_transition.get(current_phase)
        
        if not info:
            return

        current_turn = self.tracker.get_phase_turn(item.sender_id)
        max_turn = info["max_turns"]
        
        should_transition = False
        
        if current_turn >= max_turn:
            should_transition = True
        elif current_turn >= info["min_turns"]:
            should_transition = await self.phase_check(item)
        if not should_transition:
            return
        
        next_phase = info["next"]
        # print(f"phase: {current_phase} to {next_phase}")
        
        # phase 전환 로그를 저장할 경로 준비
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        log_time = self.tracker.get_time(sender_id)
        if not log_time:
            log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.tracker.set_time(sender_id, log_time)

        user_dir = os.path.join(log_dir, sender_id)
        os.makedirs(user_dir, exist_ok=True)

        safe_time = log_time.replace(":", "-").replace(" ", "_")
        log_path = os.path.join(user_dir, f"{safe_time}_{sender_id}.json")

        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"history": []}

        data["history"].append({
            "speaker": "Stage change",
            "text": f"phase: {current_phase} to {next_phase}"
        })

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # tracker의 phase 상태와 phase별 history 초기화
        self.tracker.set_phase(item.sender_id, next_phase)
        self.tracker.set_phase_turn(item.sender_id, 0)
        self.tracker.reset_history_phase(item.sender_id)

        if next_phase == "cesd_start":
            self.tracker.set_chat_mode(item.sender_id, "cesd_start")
            print(self.tracker.get_phase(item.sender_id))

        return
    
    def _load_label_examples(self): # 질문의 예시를 가져와서 prompt에 넣을 수 있도록 함
        # 모델이 더 다양한 질문을 하기 위한 과거 질문 데이터 활용

        SCORE_FILE = "Counseling_KOR_2025/data/scores_output_clean.jsonl"

        label_examples = defaultdict(list)

        with open(SCORE_FILE, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                label_examples[obj["label"]].append(obj)

        # total_score 기준 정렬
        for lab in label_examples:
            label_examples[lab].sort(
                key=lambda x: x["total_score"],
                reverse=True
            )

        return label_examples

    # 질문 종류 설명
    def _labels_to_instruction(
        self,
        missing_labels: list[str],
        max_labels: int = 2,
        examples_per_label: int = 2,
    ) -> str:

        label_map = {
            "background": "the user's daily life, routines, and surrounding environment",
            "current_state": "their recent overall condition and how things have been lately",
            "cause_onset": "when and how the current difficulty began",
            "situational_context": "the specific situations or contexts in which difficulties arise",
            "emotional_cognitive": "their thoughts, interpretations, and emotional reactions",
            "impact_functional": "how this issue affects daily functioning, work, or relationships",
            "clarification": "any parts that remain unclear and need elaboration",
            "session_goal": "what the user hopes to get from this session",
        }

        selected = missing_labels[:max_labels]

        if not selected:
            return "- (no missing areas)"

        blocks = []

        for lab in selected:

            desc = label_map.get(lab, lab)

            examples = self.label_examples.get(lab, [])[:examples_per_label]

            ex_texts = [x["text"] for x in examples]

            block = f"- {desc}"

            if ex_texts:

                example_block = "\n".join(
                    f"  - {text}" for text in ex_texts
                )

                block += (
                    "\n"
                    "  Example question styles (You must NOT copy verbatim):\n"
                    f"{example_block}"
                )

            blocks.append(block)

        return "\n\n".join(blocks)
    
    
    def get_prompt(self, mode, item, expression): 
         # 현재 mode와 phase에 따라 사용할 prompt를 선택하고 필요한 정보를 채워 반환
        history_input = self.tracker.get_history(item.sender_id)[-4:]
        user_persona = self.tracker.get_persona(item.sender_id)
        
        # 상담 및 CES-D 시작 단계에서는 phase별 prompt 사용
        if mode in ("counseling", "cesd_start"):
            if self.tracker.get_phase(item.sender_id) == "exploration":
                all_labels = get_all_labels()
                missing_labels = self.tracker.get_missing_labels(item.sender_id, all_labels)
                missing_label_instruction = self._labels_to_instruction(missing_labels[:2])
                prompt = self.prompts["exploration_instruction2"].format(missing_label_instruction = missing_label_instruction, history=history_input, persona=user_persona, expression= expression)
                print(prompt)
                return prompt
            elif self.tracker.get_phase(item.sender_id) == "resolution":
                prompt = self.prompts["resolution_instruction"].format(history=history_input, persona=user_persona, expression= expression)
                return prompt
            elif self.tracker.get_phase(item.sender_id) in ("closing", "cesd_start"):
                prompt = self.prompts["closing_instruction"].format(history=history_input, persona=user_persona, expression= expression) 
                return prompt
        else: # mode == "result"
            # result 모드에서는 CES-D 점수와 결과를 기반으로 결과 prompt 생성
            prompt = self.prompts[mode]
            score, result = self.tracker.get_cesd_result(item.sender_id)
            self.tracker.set_score(item.sender_id, score)
            prompt = prompt.format(self.tracker.get_score(item.sender_id), result = result)
            
            return prompt
    
    def format_input(self, mode, item): 
        # 현재 mode에 맞게 모델 입력 형식을 구성
        text = item.text if item.text is not None else ""
        # 상담 및 CES-D 진행 단계에서는 사용자 입력 형식으로 변환
        if mode in ("counseling", "cesd_start"):
            return 'user: ' + text
        elif mode == 'cesd':
            return 'user: ' + text
        # 결과 단계에서는 저장된 CES-D 점수와 결과를 기반으로 입력 생성
        elif mode == 'result':
            score, result = self.tracker.get_cesd_result(item.sender_id)
            self.tracker.set_score(item.sender_id, score)
            return self.prompts['result_input'].format(score = self.tracker.get_score(item.sender_id), result = result)
        else:
            return text
    
    # 모델 답변을 체크하는 함수. 욕설 방지 및 closing phase에 알맞게 들어갔는지 확인한다.
    async def post_process_llm(self, item, response):
        if self.tracker.get_phase(item.sender_id) == "closing":
            inst = self.prompts["post_process_closing"].format(
            turn=self.tracker.get_phase_turn(item.sender_id),
            history=self.tracker.get_history_phase(item.sender_id),
            utt=response,
            )
            print(inst)
        else:
            inst = self.prompts["post_process_llm"].format(
            history=self.tracker.get_history_phase(item.sender_id),
            utt=response
        )
        messages = [
            {"role": "system", "content": inst},
            {"role": "user", "content": ""}
        ]

        completion = await client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=messages,
            max_tokens=256,
            temperature=0.7
        )

        return completion.choices[0].message.content.strip()
    
    def post_process_result(self, item, output):
        # result mode의 후처리 
        if not isinstance(output, str):
            output = str(output)
        text = output.strip()
        
        junk_pattern = r'^(?:\s*(?:네|예)\s*[,.]?\s*)?(?:알겠습니다|확인했습니다|좋습니다|네\s*알겠습니다)\s*[.!?。]*\s*'
        new_text = re.sub(junk_pattern, '', text)
        text = new_text.strip()
        
        if isinstance(self.tracker.get_score(item.sender_id), int):
            text = f"점수: {self.tracker.get_score(item.sender_id)} {text}".strip()
        return text
    
    ASYNC_LABEL_THRESHOLD = 0.12 
    async def generate(self, mode, item, meta): 
        # 모델 응답 생성하고 후처리까지 진행
        
        # system prompt와 user input 구성
        sys_prompt = self.get_prompt(mode, item, meta["txt"])
        user_input = self.format_input(mode, item)
        
        messages = [
            {'role': 'system',
            'content': sys_prompt
            },
            {'role': 'user',
            'content': user_input
            }
        ]
        # counseling / result / cesd_start 모드에서는 vLLM 서버 호출
        if mode in ("counseling", "result", "cesd_start"):
            payload = {
                "model": "counseling",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 256,
            }

            r = await self.vllm.post("/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            output = data["choices"][0]["message"]["content"].strip()
         # 모델 특수 토큰 제거
        output = output.split('[|assistant|]')[-1].replace('[|endofturn|]', '')
        output = self.post_process_response(item, output)
        print(f'output: {output}\n')
        
        # exploration 단계에서는 상담 주제를 label classifier로 분류하여 저장
        if self.tracker.get_phase(item.sender_id) == "exploration":
            res = classify_label(output)
            self.tracker.add_asked_label(item.sender_id, res["pred_label"])
            print(res["pred_label"])
        
        # result 모드에서는 결과 전용 후처리 수행 후 바로 반환
        if mode == "result":
            output = self.post_process_result(item, output)
            return output
        
        # 2차 후처리: GPT 기반 안전성 및 closing 적절성 검사
        output = await self.post_process_llm(item, output)
        print(f'processing 후 output: {output}\n')
        return output

    
    # 모델 응답에서 불필요한 인사말, speaker tag, 특수문자 등을 제거하는 1차 후처리 함수
    def post_process_response(self, item, response):
        if len(self.tracker.get_history(item.sender_id)) > 0 or self.tracker.get_chat_mode(item.sender_id) == 'cesd': 
            response = response.replace('안녕하세요,', '').replace('안녕하세요.', '').strip()
            response = response.replace('상담사입니다. ', '').replace('상담사 입니다.', '').strip()
            response = response.replace('counselor:', '').replace('상담사:', '').replace('Counselor', '').strip()
            response = response.replace('"', '').strip()
            response = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,!?]", "", response)

        if self.tracker.get_chat_mode(item.sender_id) == 'counseling':
            response = response.replace('안녕하세요,', '').replace('안녕하세요.', '').strip()
            return response
        else:
            return response.strip()
    
    def symptom_format(self, text, instruction):
        return {
        'text': f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {instruction}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {text}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        <|eot_id|>"""}
    
    # user의 utt에 cesd가 관련되어 있는지 판단
    def detect_symptom(self, item):
        # print("detect_symptom 함수")
        instruction = self.prompts["cesd_symptom_instruction"]
        input_text = self.symptom_format(item.text, instruction)
        inputs = self.llama_tokenizer(input_text['text'], return_tensors="pt", padding='max_length', truncation=True, max_length=1024).to('cuda:1')
        self.llama_model.set_adapter("symptom")
        with torch.no_grad():
            outputs = self.llama_model(**inputs)
            logits = outputs.logits
        assistant_index = self.llama_tokenizer.encode("assistant", add_special_tokens=False)
        start_index = (inputs["input_ids"] == assistant_index[0]).nonzero(as_tuple=True)[1][0].item() + 1
    
        assistant_logits = logits[:, start_index:, :]
        probs = torch.softmax(assistant_logits, dim=-1)  # 확률 계산

        # 상위 10개 확률을 추출
        top_probs, top_indices = torch.topk(probs[0, 1], k=10)  # 두 번째 포지션에서 상위 10개 확률과 토큰 인덱스
        
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token = self.llama_tokenizer.decode([idx]).strip()
            
            # 공백인 경우는 제외
            if token == "":
                continue
            
            # 숫자 토큰은 허용
            if token.isdigit():
                token = int(token)
                return token
            
            # 확률이 0.2 이상인 경우만 출력
            if prob >= 0.2:
                token = int(token)
                return token
        
        # 확률이 0.2 이상인 토큰이 없을 경우 None 리턴
        return 0
    
    
    def cesd_question(self, item):
        # 현재 CES-D 문항 번호에 해당하는 질문을 반환
        symptom_num = self.tracker.get_cesd_num(item.sender_id)
        question = self.prompts["cesd_item"][symptom_num]
        print(question)
        return str(question)
    
    
    def freq_format(self, text, instruction):
        # scoring 모델 입력을 chat template 형식으로 변환
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": text}
        ]
        formatted = self.llama_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    

    def score_depression(self, item):
        # 사용자 응답을 기반으로 CES-D 점수를 예측
        self.llama_model.set_adapter("freq")
        instruction = self.prompts["cesd_symptom_score"]

        formatted = self.freq_format(item.text, instruction)
        inputs = self.llama_tokenizer(formatted, return_tensors="pt").to("cuda:1")

         # 점수 생성
        with torch.no_grad():
            outputs = self.llama_model.generate(
                **inputs,
                max_new_tokens=3,          
                temperature=0.0,           
                do_sample=False,
                pad_token_id=self.llama_tokenizer.eos_token_id
            )

        # 생성 결과 디코딩
        out = self.llama_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        match = re.search(r'(-1|0|1|2|3)', out)
        score = int(match.group(1)) if match else -1

        print(f"입력: {item.text}\n→ 예측 점수: {score} (원시 출력: {out})")
        return score
    
    def maybe_update_persona_by_batch(self, sender_id: str, user_text: str):
    # persona 대상 발화가 들어올 때마다 count
        self.tracker.increase_turn(sender_id)

        # 4, 7, 10 ... 번째 persona 대상 발화 시점에
        # 직전 3개 발화로 persona 추출
        if self.tracker.get_turn(sender_id) % 3 == 1:
            persona_input = self.tracker.get_persona_input(sender_id)
            if persona_input:
                new_persona = self.personaExtractor.predict_persona(persona_input)
                self.tracker.update_persona(sender_id, new_persona)
                self.tracker.clear_persona_utts(sender_id)

        # 현재 발화는 다음 batch를 위해 저장
        self.tracker.add_persona_utt(sender_id, user_text)
    
    # 사용자 입력을 먼저 처리하고, 현재 상태에 따라 다음 동작을 결정
    def pre_chat(self, item):
        try:
            # 상담이 이미 종료된 경우
            if self.tracker.get_terminate(item.sender_id): # 상담이 끝난 경우
                return '상담이 종료되었습니다.', self.tracker.get_cesd_info(item.sender_id), False, True
            # 사용자 발화를 history에 저장
            self.tracker.insert_history(item.sender_id, item.text, "User")
        
            # CES-D scoring 단계인 경우
            if self.tracker.get_chat_mode(item.sender_id) == "dial_cesd" or self.tracker.get_chat_mode(item.sender_id) == "cesd":
                # 사용자 응답에 대한 CES-D 점수 계산
                score = self.score_depression(item)

                self.tracker.insert_cesd_info(item.sender_id, score, self.tracker.get_cesd_num(item.sender_id))
                self.tracker.set_cesd_num(item.sender_id, -1)
                self.tracker.set_history_phase(item.sender_id, "user: " + item.text)

                 # 대화 중 삽입된 CES-D 질문(dial_cesd) 처리
                if self.tracker.get_chat_mode(item.sender_id)  == "dial_cesd":
                    # dial_cesd의 경우 점수 매겼고 counseling으로 모드 바꿈
                    self.tracker.set_chat_mode(item.sender_id, "counseling")
                    self.maybe_update_persona_by_batch(item.sender_id, item.text)
                    emo = self.get_emotion(item)
        
                    # emotion, cesd info, generation 여부, 종료 여부 반환
                    return emo, self.tracker.get_cesd_info(item.sender_id), True, False

                # CESD mode
                else: 
                    # scoring 불가 처리
                    if score == -1:
                        self.tracker.set_cesd_count(item.sender_id, self.tracker.get_cesd_count(item.sender_id) + 1)
                        if self.tracker.get_cesd_count(item.sender_id) > 3:
                            self.tracker.set_cesd_count(item.sender_id, 0) 
                            # 반복 실패 시 invalid 처리
                            self.tracker.insert_cesd_info(item.sender_id, -2, self.tracker.get_cesd_num(item.sender_id))
                    
                    # 아직 CES-D가 끝나지 않은 경우
                    if self.tracker.isdone_cesd(item.sender_id) == False:
                        # 다음 질문할 문항 번호 가져오기
                        new_symptom_num = self.tracker.get_current_cesd(item.sender_id)
                        if new_symptom_num != self.tracker.get_cesd_num(item.sender_id):
                            self.tracker.set_cesd_count(item.sender_id, 0)

                        self.tracker.set_cesd_num(item.sender_id, new_symptom_num)
                        # 다음 CES-D 질문 생성
                        response = self.cesd_question(item)
                        self.tracker.insert_history(item.sender_id, response, "Counselor")
                    
                        return response, self.tracker.get_cesd_info(item.sender_id), False, False
                    
                    else: 
                        # CES-D 완료 후 result 모드로 변경
                        self.tracker.set_chat_mode(item.sender_id, "result")
                        return "result", self.tracker.get_cesd_info(item.sender_id), True, False
            
            # 대화 종료 후 CES-D 시작 단계
            elif self.tracker.get_chat_mode(item.sender_id) == "cesd_start": 
                if self.tracker.isdone_cesd(item.sender_id) == False:
                    # 첫 CES-D 질문 준비
                    self.tracker.set_cesd_num(item.sender_id, self.tracker.get_current_cesd(item.sender_id))
                    response = self.cesd_question(item)
                    
                    # 첫 질문 전 안내 문구 추가
                    if type(response) == str:
                        response = "더 나은 상담 결과를 위해, 몇 가지 질문을 여쭤보겠습니다." + response
                    self.tracker.set_chat_mode(item.sender_id, "cesd")
                    self.tracker.insert_history(item.sender_id, response, "Counselor")
                    
                    return response, self.tracker.get_cesd_info(item.sender_id), False, False
                    
            # 사용자 발화에서 우울 증상(CESD) 관련 항목 탐지
            symptom_num = self.detect_symptom(item)
            self.tracker.set_cesd_num(item.sender_id, symptom_num)

            # 현재 phase history 및 turn 수 업데이트
            self.tracker.set_history_phase(item.sender_id, "user: " + item.text)
            self.tracker.set_phase_turn(item.sender_id, self.tracker.get_phase_turn(item.sender_id) + 1)


            # 새로운 symptom이 탐지되었고 아직 score되지 않은 경우
            if 1 <= self.tracker.get_cesd_num(item.sender_id) <= 20 and  self.tracker.has_cesd_item(item.sender_id, self.tracker.get_cesd_num(item.sender_id)) == False:
                # 관련 CES-D 질문 생성
                response = self.cesd_question(item)
                # 다음 턴에서 scoring 가능하도록 mode 변경
                self.tracker.set_chat_mode(item.sender_id, "dial_cesd")
                self.tracker.insert_history(item.sender_id, response, "Counselor")
                return response, self.tracker.get_cesd_info(item.sender_id), False, False
            # 일반 counseling 모드 처리
            if self.tracker.get_chat_mode(item.sender_id) == "counseling":
                self.maybe_update_persona_by_batch(item.sender_id, item.text)
                emo = self.get_emotion(item)
                if self.tracker.get_chat_mode(item.sender_id) == "cesd_start":
                    print("cesd_start")
                    return emo, self.tracker.get_cesd_info(item.sender_id), False, False
                print("여기가 return")
                return emo, self.tracker.get_cesd_info(item.sender_id), True, False
            
        except Exception as e:
            logging.error("Error: %s", e, exc_info=True)
            print(str(e))
            return "", self.tracker.get_cesd_info(item.sender_id), False, str(e)
    