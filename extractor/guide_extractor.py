# -*- coding: utf-8 -*-
import re
import time
import coloredlogs, logging
from typing import List, Optional
from data.data_process.data_structures import PrivacyItem, ExtractionResult
from utils.text_utils import extract_sentences

class GuideExtractor:
    def __init__(self, config, kb):
        self.config = config
        self.kb = kb
        self.logger = logging.getLogger(__name__)
        # 默认接收方可配置，避免写死“开发者”
        self.default_recipient = getattr(config, "guide_default_recipient", "开发者")

        self.processed_documents = 0

        # 模板句式A：为了...，开发者收集/使用/访问/调用你的...
        self.pattern_a = re.compile(
            r"为了([^\n，。]*?)(?:[,，。]|，|。).*?(?:开发者|我们).*?"
            r"(收集|使用|访问|调用|读取).*?"
            r"你(?:选中的|选择的|挑选的|拍摄的|上传的|填写的|的)?([^，。；\n（(]+)",
            re.MULTILINE
        )

        # 模板句式B：开发者收集你的...，用于...
        self.pattern_b = re.compile(
            r"(?:开发者|我们).*?"
            r"(收集|使用|访问|调用|读取).*?"
            r"你(?:选中的|选择的|挑选的|拍摄的|上传的|填写的|的)?([^，。；\n（(]+).*?"
            r"(?:用于|以便于|为了)([^，。；\n]+)",
            re.MULTILINE
        )
    def extract_single_guide(self, text: str, app_id: Optional[str] = None) -> Optional[ExtractionResult]:
        start_time = time.time()
        start_m = re.search(r'开发者[：:]\s*', text)
        if not start_m:
            # 无锚点时回退为全文，提升召回
            self.logger.warning(f"[{app_id}] 未找到“开发者”锚点，改为全文句分。")
            start_idx = 0
        else:
            start_idx = start_m.end()
        end_m = re.search(r'\r?\n?\s*用户权益(?:[：:]|\s*:)?', text[start_idx:])
        end_idx = start_idx + end_m.start() if end_m else len(text)
        core_text = text[start_idx:end_idx].strip()
        sentences = extract_sentences(core_text)
        self.logger.info(f"[{app_id}] 从指引中提取 {len(sentences)} 个句子用于规则抽取")

        privacy_items: List[PrivacyItem] = []

        matches_a = list(self.pattern_a.finditer(core_text))
        for i, match in enumerate(matches_a):
            purpose, method, data_type = match.group(1), match.group(2), match.group(3)
            item = PrivacyItem(
                data_type=self.kb.normalize_term((data_type or "").strip()),
                purpose=self.kb.normalize_term((purpose or "").strip()),
                processing_method=(method or "").strip(),
                recipients=[self.default_recipient],
                confidence=0.95,
                source="guide_template_A",
                evidence_text=match.group(0).strip(),
                sentence_id=i
            )
            privacy_items.append(item)

        offset = len(privacy_items)
        matches_b = list(self.pattern_b.finditer(core_text))
        for j, match in enumerate(matches_b):
            method, data_type, purpose = match.group(1), match.group(2), match.group(3)
            item = PrivacyItem(
                data_type=self.kb.normalize_term((data_type or "").strip()),
                purpose=self.kb.normalize_term((purpose or "").strip()),
                processing_method=(method or "").strip(),
                recipients=[self.default_recipient],
                confidence=0.95,
                source="guide_template_B",
                evidence_text=match.group(0).strip(),
                sentence_id=offset + j
            )
            privacy_items.append(item)
        extraction_time = time.time() - start_time
        result = ExtractionResult(
            app_id=app_id or "",
            privacy_items=privacy_items,
            total_sentences=len(privacy_items),
            processed_sentences=len(privacy_items),
            extraction_time=extraction_time,
            model_calls=0
        )
        privacy_items = self._filter_empty_data_type(privacy_items)
        # 正常流程返回结果
        self.logger.info(
            f"[{app_id}] 模板化指引抽取完成: {len(privacy_items)} 个隐私项，耗时 {extraction_time:.2f}s"
        )
        return result

    def _filter_empty_data_type(self, items: List[PrivacyItem]) -> List[PrivacyItem]:
        return [item for item in items if item.data_type and item.data_type.strip()]
