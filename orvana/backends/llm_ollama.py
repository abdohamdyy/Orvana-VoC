"""Stage 2 backend: an open-source LLM served locally by Ollama (no external API, no keys)."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from ..config import settings
from ..llm_cache import LLMCache
from ..schema import Category
from . import prompts


class OllamaBackend:
    def __init__(self, model: str | None = None, prompt_version: str = "v3", fewshot_k: int = 8, workers: int = 2,
                 cache: bool = True):
        """cache=False: nothing is ever written to the disk cache (the live chat / UI: customer text
        must not be persisted). The batch pipeline keeps the cache (reproducible, resumable runs)."""
        self.model = model or settings.llm_model
        self.prompt_version = prompt_version
        self.fewshot_k = fewshot_k
        self.workers = workers
        self.cache = LLMCache() if cache else None
        self.digest = self._model_digest()
        self.name = f"ollama:{self.model}@{self.digest[:12]}:{prompt_version}"
        self.call_seconds: list[float] = []  # latency of real (non-cached) calls
        if prompt_version == "v3":
            from ..retriever import load_retriever  # only v3 needs the training data

            self.retriever = load_retriever()

    # ------------------------------------------------------------------ plumbing
    def _model_digest(self) -> str:
        tags = requests.get(f"{settings.ollama_url}/api/tags", timeout=10).json()
        for m in tags.get("models", []):
            if m["name"] == self.model or m["model"] == self.model:
                return m["digest"]
        raise RuntimeError(f"model {self.model} not pulled; run: ollama pull {self.model}")

    def _chatml(self, messages: list[dict]) -> str:
        """Build the Qwen chat template ourselves (raw mode). For Qwen3 an empty <think></think>
        block switches its reasoning off: the installed Ollama build ignores the `think: false`
        flag and a single call then took 90 s."""
        prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        prompt += "<|im_start|>assistant\n"
        if self.model.startswith("qwen3"):
            prompt += "<think>\n\n</think>\n\n"
        return prompt

    def _chat(self, messages: list[dict], schema: dict, temperature: float = 0.0, seed: int = 42,
              use_cache: bool = True) -> dict:
        """use_cache=False (the chatbot): nothing about the call is written to disk, so customer
        conversations are never persisted."""
        use_cache = use_cache and self.cache is not None
        parts = [self.digest, self.prompt_version, json.dumps(messages, ensure_ascii=False)]
        if (temperature, seed) != (0.0, 42):  # default decoding keeps the original key (existing cache stays valid)
            parts.append(f"t={temperature},s={seed}")
        key = LLMCache.key(*parts)
        if use_cache and (hit := self.cache.get(key)) is not None:
            return hit
        body = {
            "model": self.model, "prompt": self._chatml(messages), "raw": True, "stream": False, "format": schema,
            "options": {"temperature": temperature, "seed": seed, "num_ctx": 4096}, "keep_alive": "30m",
        }
        last_err = None
        for _ in range(2):  # one retry on timeout / bad JSON
            try:
                t0 = time.perf_counter()
                r = requests.post(f"{settings.ollama_url}/api/generate", json=body, timeout=settings.llm_timeout_s)
                r.raise_for_status()
                out = json.loads(r.json()["response"])
                self.call_seconds.append(time.perf_counter() - t0)
                if use_cache:
                    self.cache.set(key, out)
                return out
            except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                last_err = e
        raise RuntimeError(f"LLM call failed twice: {last_err}")

    # ------------------------------------------------------------------ v1: everything from the LLM
    def extract_all(self, text: str) -> dict:
        msgs = [{"role": "system", "content": prompts.V1_SYSTEM}, {"role": "user", "content": prompts.wrap(text)}]
        return self._chat(msgs, prompts.V1_SCHEMA)

    # ------------------------------------------------------------------ v2/v3: one category per segment
    def _segment_messages(self, segment: str) -> list[dict]:
        if self.prompt_version == "v2":
            return [{"role": "system", "content": prompts.V2_SYSTEM}, {"role": "user", "content": prompts.wrap(segment)}]
        msgs = [{"role": "system", "content": prompts.V3_SYSTEM}]
        for ex_text, ex_label in self.retriever.nearest(segment, self.fewshot_k):
            msgs.append({"role": "user", "content": prompts.wrap(ex_text)})
            msgs.append({"role": "assistant", "content": json.dumps({"category": ex_label})})
        msgs.append({"role": "user", "content": prompts.wrap(segment)})
        return msgs

    def classify_one(self, segment: str) -> tuple[Category, float]:
        out = self._chat(self._segment_messages(segment), prompts.CATEGORY_SCHEMA)
        return Category(out["category"]), 1.0  # Ollama gives no calibrated probability

    def classify(self, segments: list[str]) -> list[tuple[Category, float]]:
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.classify_one, segments))
