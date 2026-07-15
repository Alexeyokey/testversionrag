from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """Ты полезный многоязычный ассистент. Отвечай только на основании предоставленного контекста.
Если ответа нет в контексте, ответь: «Ответ отсутствует в предоставленных документах».
Отвечай на языке вопроса. Не показывай ход рассуждений. При ссылке на фрагмент указывай
его номер в формате [1], [2]. Давай краткий, законченный ответ."""

HUMAN_TEMPLATE = """Контекст:

{context}

Вопрос:

{question}

Ответь только на основании контекста."""


class TextGenerator:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        trust_remote_code: bool = True,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
        except ImportError as error:
            if "gptqmodel" in str(error).lower():
                raise RuntimeError(
                    "Выбрана AWQ-модель, для которой нужен gptqmodel. "
                    "На Windows используйте неквантованную модель, например "
                    "RAG_GENERATION_MODEL=Qwen/Qwen3.5-0.8B."
                ) from error
            raise

        self.model.eval()

    def answer(self, question: str, context: str) -> str:
        human_content = HUMAN_TEMPLATE.format(context=context, question=question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": human_content},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_ids = output_ids[0][input_length:]
        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
