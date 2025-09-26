import inflect
from dataclasses import dataclass
from tqdm import tqdm
import deepl
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from typing import List, Union
from openai import OpenAI
import os
import logging
import time
import http.client


from vllm import LLM, SamplingParams
import json

logger = logging.getLogger(__name__)


def inflect_to_plural(phrase):
    p = inflect.engine()
    tokens = phrase.split(" ")
    word_to_inflect = tokens[-1]
    plural_word = p.plural(word_to_inflect)
    return " ".join(tokens[:-1] + [plural_word])


class DeeplTranslator:
    def __init__(self, auth_key: str = None):
        auth_key = auth_key or os.environ.get("DEEPL_API_KEY", None)
        if auth_key is None:
            raise ValueError("Please provide a DeepL API key.")

        self.translator = deepl.DeepLClient(auth_key)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def translate(
        self,
        text: Union[str, List[str]],
        source_lang: str,
        target_lang: str,
        # split_sentences: str = "on",
        model_type: str = "prefer_quality_optimized",
    ):
        result = self.translator.translate_text(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            # split_sentences=split_sentences,
            model_type=model_type,
        )
        return [r.text for r in result]

    def translate_batch(self, texts):
        return [self.translate_text(text) for text in tqdm(texts, desc="Text")]


class OpenWeightTranslator:
    def __init__(self, model_name_or_path: str):
        self.llm = LLM(
            model=model_name_or_path,
            dtype="bfloat16",
            max_model_len=4096,
            enable_prefix_caching=True,
            disable_sliding_window=True,
        )

    def translate(
        self,
        texts: List[str],
        apply_chat_template: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        # guided_decoding_params = GuidedDecodingParams(
        #     regex="<it>\s\*\*(GENDERED|NEUTRAL)\*\*\s\[[^\]]+\]",
        # )
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        if apply_chat_template:
            texts = [
                self.llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_template=True,
                )
                for t in texts
            ]
        outputs = self.llm.generate(texts, sampling_params)
        output_texts = [o.outputs[0].text for o in outputs]
        return output_texts


class OpenAITranslator:

    def __init__(self, model_name: str):
        self.client = OpenAI()
        self.model_name = model_name

    def translate(
        self,
        texts: List[str],
        apply_chat_template: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        completions = list()
        for count, prompt in tqdm(
            enumerate(texts), desc="GPT API calls", total=len(texts)
        ):
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=2048,
                model=self.model_name,
            )
            translated_text = response.choices[0].message.content
            completions.append(translated_text)
            if (count + 1) % 60 == 0:
                logger.info(
                    "Completed", count + 1, "prompts. Sleeping for one minute..."
                )
                time.sleep(65)
        return completions


class TowerTranslator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = http.client.HTTPSConnection("api.widn.ai")

        self.headers = {
            "Content-Type": "application/json",
            "X-API-KEY": "k-64d8c4b8-e001-704f-2baf-0fef829f5827-s-ZtfLMrhaCz8BiYeDzn2gEoqvPzw22fhD",
        }

    def translate(
        self,
        texts: List[str],
        apply_chat_template: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
        def _translate_single(text):
            payload = json.dumps(
                {
                    "sourceText": [text],
                    "config": {
                        "model": self.model_name,
                        "sourceLocale": "en",
                        "targetLocale": "de",
                        "maxTokens": 1024,
                    },
                }
            )
            self.client.request("POST", "/v1/translate", payload, self.headers)

            data = self.client.getresponse().read()
            result = json.loads(data.decode("utf-8"))
            translated_text = result["targetText"][0]
            return translated_text

        completions = list()
        for count, prompt in tqdm(
            enumerate(texts), desc="Tower API:", total=len(texts)
        ):
            try:
                translated_text = _translate_single(prompt)
            except Exception as e:
                logger.error(f"Error translating: {e}")
                translated_text = "ERROR"

            completions.append(translated_text)
            if (count + 1) % 60 == 0:
                print("Completed", count + 1, "prompts. Sleeping for one minute...")
                time.sleep(65)

        return completions
