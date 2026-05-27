from pydantic import BaseModel
from typing import List

class Turn(BaseModel):
    role: str
    text: str

# 데모 ver (이걸 써야됨)
class Chat1_input_demo(BaseModel): 
    sender_id: str # user id
    text: str # user utterance
    image: str

class Chat1_output_demo(BaseModel):
    text: List[str] # [system utterance]
    score: str 
    is_ending: bool

# # 실증실험 ver
# class Chat1_input(BaseModel):
#     sender_id: str
#     history: List[Turn]
#     message: str
#     image: str

# class Chat1_output(BaseModel):
#     recipient_id: str
#     history: List[Turn]
#     message: str
#     done: bool
#     error: str