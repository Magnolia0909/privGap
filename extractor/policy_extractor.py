# -*- coding: utf-8 -*-
import os
import time
import coloredlogs, logging
from typing import List, Optional

from data.data_process.data_structures import PrivacyItem, ExtractionResult
from utils.text_utils import extract_sentences

from extractor.llm_extractor import LLMExtractor
from extractor.ner_extractor import NERExtractor


class PolicyExtractor:
    def __init__(self, config, kb, llm_client=None):
        self.config = config
        self.kb = kb
        self.logger = logging.getLogger(__name__)
        self.use_llm = self.config.policy_use_llm
        self.use_ner = self.config.policy_use_ner

        self.llm_extractor = (
            LLMExtractor(config, kb, llm_client)
            if self.use_llm and llm_client
            else None
        )

        self.ner_extractor = NERExtractor(config, kb) if self.use_ner else None

    def extract_single_policy(self, text: str, app_id: Optional[str] = None) -> ExtractionResult:
        if not text or not isinstance(text, str):
            raise TypeError(f"[PolicyExtractor]输入文本无效: {type(text)}")
        start_time = time.time()
        sentences = extract_sentences(text)
        if not sentences:
            raise ValueError(f"[{app_id}]无法从文本中提取有效句子。")
        privacy_items: List[PrivacyItem] = []
        model_calls = 0
        matched_indices = set()
            
        if self.ner_extractor:
            ner_input = [(i, sent) for i, sent in enumerate(sentences) if i not in matched_indices]

            if ner_input:
                ner_sentences = [sent for _, sent in ner_input]
                ner_items, _ = self.ner_extractor.extract(ner_sentences, app_id)
                # 重新映射 sentence_id 到原始索引
                idx_mapping = {new_idx: orig_idx for new_idx, (orig_idx, _) in enumerate(ner_input)}
                for item in ner_items:
                    if item.sentence_id >= 0 and item.sentence_id in idx_mapping:
                        item.sentence_id = idx_mapping[item.sentence_id]
                privacy_items.extend(ner_items)
                matched_indices.update(item.sentence_id for item in ner_items if item.sentence_id >= 0)
                self.logger.info(f"[{app_id}] NERExtractor 抽取 {len(ner_items)} 条。")
        
        if self.llm_extractor:
            remaining = [s for i, s in enumerate(sentences) if i not in matched_indices]
            if remaining:
                max_attempts = 3
                last_err = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        llm_items, llm_calls = self.llm_extractor.extract(remaining, app_id)
                        privacy_items.extend(llm_items)
                        model_calls += llm_calls
                        self.logger.info(f"[{app_id}] LLMExtractor抽取了{len(llm_items)}条。")
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        self.logger.error(
                            f"[{app_id}] LLMExtractor抽取失败(第{attempt}次), 已重试: {e}"
                        )
                if last_err is not None:
                    self.logger.error(f"[{app_id}] LLMExtractor抽取失败, 已跳过LLM阶段:{last_err}")
                    self._record_llm_failure(app_id, str(last_err))
        
        elapsed = time.time() - start_time
        # 统一规范化 + 去重
        normalized_items = [self._normalize_item(item) for item in privacy_items]
        deduped_items = self._dedup_items(normalized_items)
        deduped_items = self._filter_empty_data_type(deduped_items)

        result = ExtractionResult(
            app_id=app_id or "",
            privacy_items=deduped_items,
            total_sentences=len(sentences),
            processed_sentences=len(deduped_items),
            extraction_time=elapsed,
            model_calls=model_calls
        )
        self.logger.info(
            f"[{app_id}] Policy 抽取完成：共 {len(deduped_items)} 条隐私项,"
            f"句子数 {len(sentences)}, 耗时 {elapsed:.2f}s, LLM 调用 {model_calls} 次。"
        )
        return result

    def _normalize_item(self, item: PrivacyItem) -> PrivacyItem:
        item.data_type = self.kb.normalize_term(item.data_type.strip()) if item.data_type else ""
        item.purpose = self.kb.normalize_term(item.purpose.strip()) if item.purpose else ""
        if item.recipients:
            uniq = []
            for r in item.recipients:
                if r and r not in uniq:
                    uniq.append(r)
            item.recipients = uniq
        return item

    def _dedup_items(self, items: List[PrivacyItem]) -> List[PrivacyItem]:
        dedup = {}
        for it in items:
            key = (
                it.data_type,
                it.purpose,
                it.processing_method,
                tuple(it.recipients),
                it.retention_period,
                it.legal_basis,
            )
            if key not in dedup:
                dedup[key] = it
            else:
                # 取更高置信度，保留较早证据
                if it.confidence > dedup[key].confidence:
                    dedup[key] = it
        return list(dedup.values())

    def _filter_empty_data_type(self, items: List[PrivacyItem]) -> List[PrivacyItem]:
        return [item for item in items if item.data_type and item.data_type.strip()]

    def _record_llm_failure(self, app_id: Optional[str], error_msg: str):
        backend = getattr(self.config, "llm_backend", "unknown")
        out_dir = getattr(self.config, "OUTPUT_DIR", "")
        if not out_dir:
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "llm_extraction_failures.jsonl")
            record = {
                "app_id": app_id or "",
                "llm_backend": backend,
                "error": error_msg,
            }
            with open(path, "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # 避免因为记录失败影响主流程
            pass
