import os
import pytorch_lightning as pl
from transformers import BartForConditionalGeneration, AutoTokenizer
import argparse

class LitModel(pl.LightningModule):
    def __init__(self, learning_rate, tokenizer, model, total_steps=0, context_window=5):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate
        self.test_loss = []
        self.total_steps = total_steps
        self.context_window = context_window

class PersonaExtractor:
    _instance = None  # 싱글톤 인스턴스 저장

    def __new__(cls, model_dir_or_name="gogamza/kobart-base-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(model_dir_or_name)
        return cls._instance

    def _initialize(self, model_dir_or_name):
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
        bart_model = BartForConditionalGeneration.from_pretrained(model_dir_or_name)

        # Add special tokens
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["<persona>", "<No persona>"]})
        bart_model.resize_token_embeddings(len(self.tokenizer))

        self.model = LitModel.load_from_checkpoint(
            "Counseling_KOR_2025/models/persona_no_history.ckpt",
            learning_rate=1e-5,
            tokenizer=self.tokenizer,
            model=bart_model,
            map_location = "cpu"
        )
        self.model.to('cuda:1')
        self.model.eval()
        print("Model and tokenizer loaded successfully!")

    def predict_persona(self, input_text, eval_beams=5, max_new_tokens=128):

        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to('cuda:1')

        # Generate persona text
        outputs = self.model.model.generate(
            inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            num_beams=eval_beams,
            early_stopping=False,
            no_repeat_ngram_size=2,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2,
            temperature=1.0
        )

        generated_persona = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_persona = generated_persona.replace("鈐", "").strip()

        return generated_persona if len(generated_persona.strip()) > 10 else ""

if __name__ == '__main__':
    PersonaExtractor()  # 한 번만 초기화
