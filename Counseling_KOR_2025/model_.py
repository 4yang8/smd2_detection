from dotenv import load_dotenv
load_dotenv()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# os.environ["DEEPSPEED_COMM_BACKEND"] = "nccl"
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
    future: asyncio.Future # 아직 끝나지 않은 비동기 작업의 결과를 담음

class CounselingChat:
    # 모델 불러오기 prompt 불러오기 등
    def __init__(self, #일단 init 완료
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
        # 이것도 실행해야 해요!
        self.vllm = httpx.AsyncClient(
            base_url = "http://141.223.163.135:9000/v1",
            timeout = 60.0
        )
        with open('Counseling_KOR_2025/data/counsel_prompt.yaml') as file:
            self.prompts = yaml.load(file, Loader=yaml.FullLoader)

        
    def get_emotion(self, item):
        # print("get emotion 함수")
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
        if self.batch_task is None or self.batch_task.done():
            self.batch_task = asyncio.create_task(self._batch_worker())
        
    async def submit_to_batch(self, item, meta):
        loop = asyncio.get_running_loop()      
        fut = loop.create_future()
        await self.batch_queue.put(BatchItem(item = item, meta = meta, future = fut))
        # print(f"DEBUG: {item.sender_id} 가 큐에 들어갔습니다. 이제 루프는 자유롭습니다.")
        return await fut
    
    async def _batch_worker(self):
        MAX_BATCH_SIZE = 2
        MAX_WAIT_TIME = 0.1
        
        while True:
            first = await self.batch_queue.get()
            batch = [first]

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

            # ✅ result 모드가 아닌 것만 phase 계산
            non_result_reqs = []
            for req in batch:
                mode = self.tracker.get_chat_mode(req.item.sender_id)
                if mode != "result":
                    non_result_reqs.append(req)

            if non_result_reqs:
                await asyncio.gather(*[self.get_phase(req.item) for req in non_result_reqs])

            gen_results = await asyncio.gather(
                *[self.generate(self.tracker.get_chat_mode(req.item.sender_id), req.item, req.meta) for req in batch]
            )

            for req, out in zip(batch, gen_results):
                sender_id = req.item.sender_id
                mode = self.tracker.get_chat_mode(sender_id)

                final_text = out
                self.tracker.insert_history(sender_id, final_text, "Counselor")
                self.tracker.set_history_phase(sender_id, "Counselor: " + final_text)

                if mode == "result":
                    self.tracker.set_terminate(sender_id)

                req.future.set_result(out)

    # 현재 상담의 어느 phase인지 체크함
    async def phase_check(self, item) -> bool:
        # print("phase_check 함수")
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
        # print(m)
        return bool(m and m.group(1) == "YES")
    
    # phase 분기함수
    async def get_phase(self, item):
        phase_transition = {
            "exploration": {"next": "resolution", "min_turns": 5, "max_turns": 7},
            "resolution": {"next": "closing", "min_turns": 3, "max_turns": 6},
            "closing": {"next": "cesd_start", "min_turns": 1, "max_turns": 1}
        }

        # print("get_phase 함수")

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
        

        self.tracker.set_phase(item.sender_id, next_phase)
        self.tracker.set_phase_turn(item.sender_id, 0)
        self.tracker.reset_history_phase(item.sender_id)

        if next_phase == "cesd_start":
            self.tracker.set_chat_mode(item.sender_id, "cesd_start")
            print(self.tracker.get_phase(item.sender_id))

        return
    
    # 질문의 예시를 가져와서 prompt에 넣을 수 있도록 함
    def _load_label_examples(self):

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
    
    # 모드, phase마다 prompt를 다르게 return하는 함수
    def get_prompt(self, mode, item, expression): 
        # print("get_prompt 함수")
        # print(self.tracker.get_phase(item.sender_id))
        history_input = self.tracker.get_history(item.sender_id)[-4:]
        user_persona = self.tracker.get_persona(item.sender_id)
       
        # print("get prompt 안의 mode")
        # print(mode)
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
            print("mode가 result")
            prompt = self.prompts[mode]
            score, result = self.tracker.get_cesd_result(item.sender_id)
            self.tracker.set_score(item.sender_id, score)
            prompt = prompt.format(self.tracker.get_score(item.sender_id), result = result)
            
            return prompt
    
    def format_input(self, mode, item): # input을 주는 함수 generate를 위해 쓰임
        #print("format_input")
        #print(item.text)
        text = item.text if item.text is not None else ""
        if mode in ("counseling", "cesd_start"):
            return 'user: ' + text
        elif mode == 'cesd':
            return 'user: ' + text
        elif mode == 'result':
            score, result = self.tracker.get_cesd_result(item.sender_id)
            self.tracker.set_score(item.sender_id, score)
            return self.prompts['result_input'].format(score = self.tracker.get_score(item.sender_id), result = result)
        else:
            return text
    
    # 모델 답변을 체크하는 함수. 욕설 방지 및 closing에 알맞게 들어갔는지 확인한다.
    async def post_process_llm(self, item, response):
        #print("post_process_llm 함수")
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
        #print("post_process_result")
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
        # print("generate 함수")
        # print("mode:",mode)
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
        # print("mode:", mode)
        
        if mode in ("counseling", "result", "cesd_start"):
            # print("counseling이나 result입니다.")
            payload = {
                "model": "counseling",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 256,
            }
            # print(messages)
            # print("[DEBUG] vllm request start", item.sender_id)
            r = await self.vllm.post("/chat/completions", json=payload)
            # print("[DEBUG] vllm response arrived", item.sender_id, r.status_code)
            # print("----생성 결과-----")
            # print(r.text)
            r.raise_for_status()
            data = r.json()
            # print("[DEBUG] vllm json parsed", item.sender_id)
            output = data["choices"][0]["message"]["content"].strip()
        
        output = output.split('[|assistant|]')[-1].replace('[|endofturn|]', '')
        print(f'output: {output}\n')
        
        if self.tracker.get_phase(item.sender_id) == "exploration":
            res = classify_label(output)
            self.tracker.add_asked_label(item.sender_id, res["pred_label"])
            print(res["pred_label"])
        
        if mode == "result":
            output = self.post_process_result(item, output)
            return output
        
        output = await self.post_process_llm(item, output)
        print(f'processing 후 output: {output}\n')
        return output

    
    # def post_process_response(self, item, response):
    #     if len(self.tracker.get_history(item.sender_id)) > 0 or self.tracker.get_chat_mode(item.sender_id) == 'cesd': 
    #         response = response.replace('안녕하세요,', '').replace('안녕하세요.', '').strip()
    #         response = response.replace('상담사입니다. ', '').replace('상담사 입니다.', '').strip()
    #         response = response.replace('counselor:', '').replace('상담사:', '').replace('Counselor', '').strip()
    #         response = response.replace('"', '').strip()
    #         response = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,!?]", "", response)

    #     if self.tracker.get_chat_mode(item.sender_id) == 'counseling':
    #         response = response.replace('안녕하세요,', '').replace('안녕하세요.', '').strip()
    #         return response
    #     else:
    #         return response.strip() 
    
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
    
    # 관련 cesd 질문을 return
    def cesd_question(self, item):
        symptom_num = self.tracker.get_cesd_num(item.sender_id)
        question = self.prompts["cesd_item"][symptom_num]
        print(question)
        # new_history = item.history + [Turn(role='Counselor', text = question)]
        return str(question)
    
    
    def freq_format(self, text, instruction):
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
    # scoring을 해야 할 차례일 때 부르는 함수
    def score_depression(self, item):
        self.llama_model.set_adapter("freq")
        instruction = self.prompts["cesd_symptom_score"]

        formatted = self.freq_format(item.text, instruction)
        inputs = self.llama_tokenizer(formatted, return_tensors="pt").to("cuda:1")

        with torch.no_grad():
            outputs = self.llama_model.generate(
                **inputs,
                max_new_tokens=3,          
                temperature=0.0,           
                do_sample=False,
                pad_token_id=self.llama_tokenizer.eos_token_id
            )

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
    
    def pre_chat(self, item):
        try:
            if self.tracker.get_terminate(item.sender_id): # 상담이 끝난 경우
                return '상담이 종료되었습니다.', self.tracker.get_cesd_info(item.sender_id), False, True
            self.tracker.insert_history(item.sender_id, item.text, "User")
            
            # print("###############################################")
            # print("chat 안의 mode: " + self.tracker.get_chat_mode(item.sender_id))
            
            if self.tracker.get_chat_mode(item.sender_id) == "dial_cesd" or self.tracker.get_chat_mode(item.sender_id) == "cesd":
                # 그렇다면 user응답에 대한 scoring 진행
                score = self.score_depression(item)
                # print(item.text)
                # print(score)
                self.tracker.insert_cesd_info(item.sender_id, score, self.tracker.get_cesd_num(item.sender_id))
                self.tracker.set_cesd_num(item.sender_id, -1)
                self.tracker.set_history_phase(item.sender_id, "user: " + item.text)

                if self.tracker.get_chat_mode(item.sender_id)  == "dial_cesd":
                    # dial_cesd의 경우 점수 매겼고 counseling으로 모드 바꿈. 생성만 더 하면 됨. + tracker history setting, phase history setting
                    self.tracker.set_chat_mode(item.sender_id, "counseling")
                    self.maybe_update_persona_by_batch(item.sender_id, item.text)
                    emo = self.get_emotion(item)
        
                    # 순서대로 emo, cesd info, exaone 호출?, is ending
                    return emo, self.tracker.get_cesd_info(item.sender_id), True, False
      
                else: ##mode가 그냥 cesd
                    if score == -1:
                        self.tracker.set_cesd_count(item.sender_id, self.tracker.get_cesd_count(item.sender_id) + 1)
                        if self.tracker.get_cesd_count(item.sender_id) > 3:
                            self.tracker.set_cesd_count(item.sender_id, 0) 
                            self.tracker.insert_cesd_info(item.sender_id, -2, self.tracker.get_cesd_num(item.sender_id))
                    if self.tracker.isdone_cesd(item.sender_id) == False:
                    # 물어봐야 하는 항목 가져오기
                        new_symptom_num = self.tracker.get_current_cesd(item.sender_id)
                        if new_symptom_num != self.tracker.get_cesd_num(item.sender_id):
                            self.tracker.set_cesd_count(item.sender_id, 0)

                        self.tracker.set_cesd_num(item.sender_id, new_symptom_num)
                        # 질문
                        response = self.cesd_question(item)
                        self.tracker.insert_history(item.sender_id, response, "Counselor")
                        # response, cesd info, exaone 호출, is ending
                        return response, self.tracker.get_cesd_info(item.sender_id), False, False
                    
                    else: 
                        print("result일 때 여기")
                        self.tracker.set_chat_mode(item.sender_id, "result")
                        return "result", self.tracker.get_cesd_info(item.sender_id), True, False
                    
            elif self.tracker.get_chat_mode(item.sender_id) == "cesd_start": # mode가 cesd start (대화 중 한 번). 바꾸는 것은 phase가 바뀌며 바꿈. 
                if self.tracker.isdone_cesd(item.sender_id) == False:
                    # 물어봐야 하는 항목 가져오기
                    self.tracker.set_cesd_num(item.sender_id, self.tracker.get_current_cesd(item.sender_id))
                    # 질문
                    response = self.cesd_question(item)
                    if type(response) == str:
                        response = "더 나은 상담 결과를 위해, 몇 가지 질문을 여쭤보겠습니다." + response
                    self.tracker.set_chat_mode(item.sender_id, "cesd")
                    self.tracker.insert_history(item.sender_id, response, "Counselor")
                    
                    return response, self.tracker.get_cesd_info(item.sender_id), False, False
                    
            # 여기서 symptom 검사
            symptom_num = self.detect_symptom(item)
            self.tracker.set_cesd_num(item.sender_id, symptom_num)
            # print("symptom setting 완료")

            self.tracker.set_history_phase(item.sender_id, "user: " + item.text)
            self.tracker.set_phase_turn(item.sender_id, self.tracker.get_phase_turn(item.sender_id) + 1)
            # print("========== history test ==============")

            # symptom_num이 검출 + 이미 score된 항목이 아닌 경우
            if 1 <= self.tracker.get_cesd_num(item.sender_id) <= 20 and  self.tracker.has_cesd_item(item.sender_id, self.tracker.get_cesd_num(item.sender_id)) == False:
                # 그 symptom_num에 대한 질문을 하고
                response = self.cesd_question(item)
                # 다음 user 응답에 대해 scoring 할 수 있도록 모드를 바꿈
                self.tracker.set_chat_mode(item.sender_id, "dial_cesd")
                self.tracker.insert_history(item.sender_id, response, "Counselor")
                return response, self.tracker.get_cesd_info(item.sender_id), False, False
            
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
        

    # 아래는 batch 처리 이전 사용했던 함수
    # def chat(self, item): # return final_response, score, done, error (이렇게 return 해야 함. 화면에 최종 띄울 것을 보내는 부분)
    #     print("start")
    #     print(item.sender_id)
    #     try:
    #         if self.tracker.get_terminate(item.sender_id): # 상담이 끝난 경우
    #             return '상담이 종료되었습니다.', self.tracker.get_cesd_info(item.sender_id), True, ""
    #         self.tracker.insert_history(item.sender_id, item.text, "User")
    #         # new_history = item.history + [Turn(role='user', text=item.text)] # add user utterance to history 

    #         print("###############################################")
    #         print("chat 안의 mode: " + self.tracker.get_chat_mode(item.sender_id))
    #         #print(self.tracker.get_history_phase(item.sender_id))
    #         #print(self.tracker.get_phase_turn(item.sender_id))
            
    #         if self.tracker.get_chat_mode(item.sender_id) == "dial_cesd" or self.tracker.get_chat_mode(item.sender_id) == "cesd":
    #             # 그렇다면 user응답에 대한 scoring 진행
    #             score = self.score_depression(item)
    #             #print(item.text)
    #             #print(score)
    #             self.tracker.insert_cesd_info(item.sender_id, score, self.tracker.get_cesd_num(item.sender_id))
    #             self.tracker.set_cesd_num(item.sender_id, -1)
    #             self.tracker.set_history_phase(item.sender_id, "user: " + item.text)

    #             if self.tracker.get_chat_mode(item.sender_id)  == "dial_cesd":
    #                 self.tracker.set_chat_mode(item.sender_id, "counseling")
    #                 #print("we are here!")
    #                 response = self.generate(self.tracker.get_chat_mode(item.sender_id), item)
    #                 #print("response 생성")
    #                 response = self.post_process_response(item, response)
      
    #                 self.tracker.set_history_phase(item.sender_id, "Counselor: " + response)
    #                 self.tracker.insert_history(item.sender_id, response, "Counselor")
    #                 print(f'final: {response}')
    #                 #print(f'new_history in dial_cesd: {self.tracker.get_history(item.sender_id)}')
    #                 return response, self.tracker.get_cesd_info(item.sender_id), False, ''
    #             else: ##mode가 그냥 cesd
    #                 if score == -1:
    #                     self.tracker.set_cesd_count(item.sender_id, self.tracker.get_cesd_count(item.sender_id) + 1)
    #                     if self.tracker.get_cesd_count(item.sender_id) > 3:
    #                         self.tracker.set_cesd_count(item.sender_id, 0) 
    #                         self.tracker.insert_cesd_info(item.sender_id, -2, self.tracker.get_cesd_num(item.sender_id))
    #                 if self.tracker.isdone_cesd(item.sender_id) == False:
    #                 # 물어봐야 하는 항목 가져오기
    #                     new_symptom_num = self.tracker.get_current_cesd(item.sender_id)
    #                     if new_symptom_num != self.tracker.get_cesd_num(item.sender_id):
    #                         self.tracker.set_cesd_count(item.sender_id, 0)

    #                     self.tracker.set_cesd_num(item.sender_id, new_symptom_num)
    #                     # 질문
    #                     response = self.cesd_question(item)
    #                     self.tracFker.insert_history(item.sender_id, response, "Counselor")
    #                     #print(f'new_history in cesd: {self.tracker.get_history(item.sender_id)}')
    #                     return response, self.tracker.get_cesd_info(item.sender_id), False, ''
    #                 else: 
    #                     final_response = self.generate('result', item)
    #                     if isinstance(self.tracker.get_score(item.sender_id), int):  # self.score가 int 타입인지 확인
    #                         self.tracker.set_score(item.sender_id, str(self.tracker.get_score(item.sender_id)))
    #                     final_response = f"점수: {self.tracker.get_score(item.sender_id)} {final_response}"
    #                     #print("score: " + self.tracker.get_score(item.sender_id))
    #                     self.tracker.set_terminate(item.sender_id)
    #                     self.tracker.insert_history(item.sender_id, final_response, "Counselor")
    #                     # print('cesd Done! Result: '+self.tracker.get_cesd_result(item.sender_id))
    #                     #print(f'new_history in result: {self.tracker.get_history(item.sender_id)}')
    #                     return final_response, self.tracker.get_cesd_info(item.sender_id), True, ''
                    
    #         elif self.tracker.get_chat_mode(item.sender_id) == "cesd_start": # mode가 cesd start (대화 중 한 번). 바꾸는 것은 phase가 바뀌며 바꿈. 
    #             if self.tracker.isdone_cesd(item.sender_id) == False:
    #                 # 물어봐야 하는 항목 가져오기
    #                 self.tracker.set_cesd_num(item.sender_id, self.tracker.get_current_cesd(item.sender_id))
    #                 # 질문
    #                 response = self.cesd_question(item)
    #                 if type(response) == str:
    #                     response = "더 나은 상담 결과를 위해, 몇 가지 질문을 여쭤보겠습니다." + response
    #                 self.tracker.set_chat_mode(item.sender_id, "cesd")
    #                 self.tracker.insert_history(item.sender_id, response, "Counselor")
    #                 #print(f'new_history in cesd start: {self.tracker.get_history(item.sender_id)}')
    #                 return response, self.tracker.get_cesd_info(item.sender_id), False, ''
                    
    #         # 여기서 symptom 검사
    #         symptom_num = self.detect_symptom(item)
    #         self.tracker.set_cesd_num(item.sender_id, symptom_num)
    #         print("symptom setting 완료")

    #         self.tracker.set_history_phase(item.sender_id, "user: " + item.text)
    #         self.tracker.set_phase_turn(item.sender_id, self.tracker.get_phase_turn(item.sender_id) + 1)
    #         print("========== history test ==============")
    #         #print(self.tracker.get_history_phase(item.sender_id))
    #         # print(self.tracker.has_cesd_item(item.sender_id, self.symptom_num))
    #         # symptom_num이 검출 + 이미 score된 항목이 아닌 경우
    #         #print(self.tracker.get_cesd_info(item.sender_id))
    #         if 1 <= self.tracker.get_cesd_num(item.sender_id) <= 20 and  self.tracker.has_cesd_item(item.sender_id, self.tracker.get_cesd_num(item.sender_id)) == False:
    #             # 그 symptom_num에 대한 질문을 하고
    #             response = self.cesd_question(item)
    #             # 다음 user 응답에 대해 scoring 할 수 있도록 모드를 바꿈
    #             self.tracker.set_chat_mode(item.sender_id, "dial_cesd")
    #             #print(f'new_history in dial_cesd 진입: {self.tracker.get_history(item.sender_id)}')
    #             self.tracker.insert_history(item.sender_id, response, "Counselor")
    #             return response, self.tracker.get_cesd_info(item.sender_id), False, ''
            
    #         new_persona = self.personaExtractor.predict_persona(item.text)
    #         self.tracker.update_persona(item.sender_id, new_persona)
                
    #         #print("persona: ", self.tracker.get_persona(item.sender_id))
            
    #         if self.tracker.get_chat_mode(item.sender_id) == "counseling":
    #             response = self.generate(self.tracker.get_chat_mode(item.sender_id), item)
    #             #print("response 생성")
    #             response = self.post_process_response(item, response)
    #             # new_history = item.history + [Turn(role='Counselor', text = response)]
    #             self.tracker.set_history_phase(item.sender_id, "Counselor: " + response)
    #             print(f'final: {response}')
    #             self.tracker.insert_history(item.sender_id, response, "Counselor")
    #             #print(f'new_history in counseling: {self.tracker.get_history(item.sender_id)}')
    #             return response, self.tracker.get_cesd_info(item.sender_id), False, ""
        
                
    #     except Exception as e:
    #         logging.error("Error: %s", e, exc_info=True)
    #         print(str(e))
    #         return "", self.tracker.get_cesd_info(item.sender_id), False, str(e)
