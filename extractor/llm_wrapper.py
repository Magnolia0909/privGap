# -*- coding: utf-8 -*-
import os
import re
import time
import coloredlogs, logging
import requests
from typing import Optional

from utils.llm_cache import LLMCache

class LLMWrapper:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.backend = self.config.llm_backend
        self.temperature = self.config.llm_temperature
        self.top_p = self.config.llm_top_p
        self.max_tokens = self.config.llm_max_token
        self.timeout = self.config.llm_timeout
        self.max_retries = self.config.llm_max_retries

        # 初始化缓存
        cache_enabled = self.config.llm_cache_enabled
        cache_dir = self.config.llm_cache_dir
        self.cache = LLMCache(cache_dir, enabled=cache_enabled)

        if self.backend == "ollama":
            self.model_name = self.config.llm_model_name_ollama
            self.api_key = self.config.llm_api_key_ollama
            self.api_url = self.config.llm_api_ollama

        elif self.backend == "deepseek":
            self.model_name = self.config.llm_model_name_deepseek
            self.api_key = self.config.llm_api_key_deepseek
            self.api_url = self.config.llm_api_deepseek
        
        elif self.backend == "qwen":
            self.model_name = self.config.llm_model_name_qwen
            self.api_key = self.config.llm_api_key_qwen
            self.api_url = self.config.llm_api_qwen
            
        elif self.backend == "gemini":
            self.model_name = self.config.llm_model_name_gemini
            self.api_key = self.config.llm_api_key_gemini
            self.api_url = self.config.llm_api_gemini

        elif self.backend == "claude":
            self.model_name = self.config.llm_model_name_claude
            self.api_key = self.config.llm_api_key_claude
            self.api_url = self.config.llm_api_claude
        else:
            raise ValueError(f"[LLMWrapper] 不支持的 LLM 后端: {self.backend}")
        
        if not self.model_name:
            raise ValueError("[LLMWrapper] 配置错误：llm_model_name 不能为空。")
        if self.backend in ("deepseek", "qwen", "openai", "gemini") and not self.api_key:
            raise ValueError(f"[LLMWrapper] 使用 {self.backend} 时必须提供 llm_api_key。")

    def chat(self, prompt: str, app_id: Optional[str] = None, task_type: str = "default") -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise TypeError("[LLMWrapper] prompt 必须是非空字符串。")

        # 尝试从缓存获取
        cached_response = self.cache.get(
            prompt=prompt,
            model_name=self.model_name,
            task_type=task_type,
            temperature=self.temperature,
            top_p=self.top_p
        )

        if cached_response is not None:
            self.logger.info(f"[{app_id}] 使用缓存响应 [{task_type}]: 模型={self.model_name}")
            return cached_response.strip()

        # 缓存未命中，调用API
        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()
                response_text = self._send_request(prompt)
                elapsed = time.time() - start_time
                self.logger.info(
                    f"[{app_id}] LLMWrapper 调用成功 [{task_type}]: 模型={self.model_name}, 耗时={elapsed:.2f}s"
                )

                # 保存到缓存
                self.cache.set(
                    prompt=prompt,
                    response=response_text,
                    model_name=self.model_name,
                    task_type=task_type,
                    temperature=self.temperature,
                    top_p=self.top_p
                )

                return response_text.strip()
            except Exception as e:
                self.logger.error(f"[{app_id}] 第 {attempt} 次模型调用失败: {e}")
                if attempt == self.max_retries:
                    raise RuntimeError(f"[LLMWrapper] 模型调用失败（已重试 {self.max_retries} 次）: {e}")
                time.sleep(1)

    def _send_request(self, prompt: str) -> str:
        if self.backend == "ollama":
            payload = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,  # 添加这个参数，禁用流式输出
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                }
            }
            resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()  # 检查 HTTP 错误
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text}")
            try:
                data = resp.json()
                return data.get("response", "")
            except Exception as e:
                raise ValueError(f"Ollama 返回解析失败: {e}")

        elif self.backend in ("deepseek", "openai", "gemini", "claude"):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }
            # 使用 session 提高连接稳定性
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                max_retries=3,
                pool_connections=10,
                pool_maxsize=10
            )
            session.mount('https://', adapter)
            session.mount('http://', adapter)

            resp = session.post(
                self.api_url,
                headers=headers,
                json=payload,  # 直接使用 json 参数
                timeout=(30, self.timeout)  # (连接超时, 读取超时)
            )
            resp.raise_for_status()  # 检查 HTTP 错误

            data = resp.json()
            if "choices" not in data or not data["choices"]:
                raise ValueError(f"响应格式错误: {data}")
            content = data["choices"][0]["message"]["content"]
            if self.backend == "deepseek" and "<think>" in content:
                if "<think>" in content:
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                
                    # 去除 Markdown 代码块标记 ```json ... ```
                    content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
                    content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)

            return content

        elif self.backend == "qwen":
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                max_retries=3,
                pool_connections=10,
                pool_maxsize=10,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            resp = session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=(10, self.timeout),
            )
            resp.raise_for_status()

            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"Qwen 返回格式异常: {data}")
            return choices[0]["message"]["content"]

        else:
            raise NotImplementedError(f"尚未支持的后端: {self.backend}")

    def get_cache_stats(self):
        """获取缓存统计信息"""
        return self.cache.get_stats()

    def print_cache_stats(self):
        """打印缓存统计信息"""
        self.cache.print_stats()
