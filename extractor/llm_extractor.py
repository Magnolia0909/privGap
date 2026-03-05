# -*- coding: utf-8 -*-
import json
import re
import os
import datetime
import coloredlogs, logging
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json_repair
from data.data_process.data_structures import PrivacyItem
class LLMExtractor:
    def __init__(self, config, kb, llm_client):
        self.config = config
        self.kb = kb
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)
        if self.llm is None or not hasattr(self.llm, "chat"):
            raise ValueError("[LLMExtractor] llm_client未提供或不支持 generate(prompt, **kwargs)。")
        self.batch_size = self.config.llm_batch_size
        self.max_workers = max(1, getattr(self.config, "llm_parallel_workers", 1))

        # 中间过程记录配置
        self.save_intermediate = getattr(self.config, "save_intermediate_data", False)
        self.intermediate_dir = getattr(self.config, "INTERMEDIATE_DIR", None)
        if self.save_intermediate and self.intermediate_dir:
            os.makedirs(self.intermediate_dir, exist_ok=True)

    def _save_intermediate_log(self, app_id: str, batch_start: int, prompt: str, raw_response: str, parsed_items: List[PrivacyItem]):
        """保存中间过程到文件"""
        if not self.save_intermediate or not self.intermediate_dir:
            return
            
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{app_id}_batch_{batch_start}_{timestamp}.json"
            filepath = os.path.join(self.intermediate_dir, filename)
            
            log_data = {
                "app_id": app_id,
                "batch_start_index": batch_start,
                "timestamp": timestamp,
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed_items": [item.to_dict() if hasattr(item, "to_dict") else str(item) for item in parsed_items]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.warning(f"[{app_id}] 保存中间过程失败: {e}")
    def extract(self, sentences: List[str], app_id: str) -> Tuple[List[PrivacyItem], int]:
        if not self.llm:
            return [], 0
        all_items: List[PrivacyItem] = []
        model_calls = 0
        batches = []
        for i in range(0, len(sentences), self.batch_size):
            batches.append((i, sentences[i:i + self.batch_size]))

        if self.max_workers > 1 and len(batches) > 1:
            # 并行按 batch 调用 LLM（IO 密集，线程足够）
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_start = {
                    executor.submit(self._process_batch, start, batch, app_id): start
                    for start, batch in batches
                }
                for future in as_completed(future_to_start):
                    items, calls = future.result()
                    all_items.extend(items)
                    model_calls += calls
        else:
            # 顺序执行
            for start, batch in batches:
                items, calls = self._process_batch(start, batch, app_id)
                all_items.extend(items)
                model_calls += calls

        if not all_items:
            self.logger.warning(
                f"[{app_id}] LLMExtractor: 合法运行，但未抽取到任何隐私项（句子数={len(sentences)}）。"
            )
            return [], model_calls
        self.logger.info(f"[{app_id}] LLMExtractor 严格抽取完成：{len(all_items)} 条。")
        return all_items, model_calls

    def _process_batch(self, start: int, batch: List[str], app_id: str) -> Tuple[List[PrivacyItem], int]:
        indexed_batch = [{"idx": idx, "text": text} for idx, text in enumerate(batch)]
        raw, prompt = self._call_llm(batch, app_id)
        triples_per_sentence = self._parse_and_validate_output(
            raw,
            expected_indices=[entry["idx"] for entry in indexed_batch],
            app_id=app_id
        )
        triples_per_sentence = self._split_merged_triples(triples_per_sentence)

        items: List[PrivacyItem] = []
        for entry in indexed_batch:
            local_idx = entry["idx"]
            triples = triples_per_sentence[local_idx]
            for t in triples:
                item = self._triple_to_item(t, entry["text"], sentence_id=start + local_idx)  # sentence_id = 全局位置
                items.append(item)
        
        # 保存中间过程
        self._save_intermediate_log(app_id, start, prompt, raw, items)
        
        return items, 1

    def _call_llm(self, indexed_batch: List[Dict[str, str]], app_id: str) -> Tuple[str, str]:
        guide = (
            "你是一个严格的隐私权限项提取器，专门从文本中提取涉及用户隐私的权限使用行为。\n\n"
            "【任务】对给定句子逐句抽取隐私权限三元组。\n"
            "输入：包含 idx(句子编号) 和 text(原句) 的JSON数组\n"
            "输出：JSON对象，键为字符串化的idx（如\"0\",\"1\"），值为该句的三元组数组。每个三元组包含4个字段：\n"
            "- data_type: 隐私权限（字符串）\n"
            "- purpose: 使用目的（字符串）\n"
            "- recipients: 接收方（字符串数组）\n"
            "- confidence: 置信度（0~1之间的小数）\n\n"

            "【confidence评分标准】\n"
            "- 0.9-1.0: 原文明确、直接的肯定陈述（如\"我们收集您的X用于Y\"）\n"
            "- 0.7-0.9: 原文较明确，但包含\"可能\"\"在某些情况下\"等修饰成分\n"
            "- 0.5-0.7: 原文表述模糊、隐晦，需要轻微推断\n"
            "- <0.5: 不确定时直接输出[]，不要猜测\n\n"

            "【输出要求】\n"
            "- 抽取的隐私权限三元组一定在原句中逐字出现，不要进行任何额外修改\n"
            "- 禁止输出除JSON以外的任何内容（禁止解释、禁止<think>、禁止Markdown代码块标记）\n"
            "- 每个idx都必须有对应的输出，如果当前句不涉及实际隐私行为，输出空数组[]\n"
            "- 确保JSON格式严格正确，可被Python json.loads()解析\n"
        )

        known_terms = []
        try:
            if hasattr(self.kb, "knowledge"):
                known_terms = list(self.kb.knowledge.keys())
        except Exception as e:
            self.logger.warning(f"[LLMExtractor] 获取已知术语失败: {e}")
        
        memory_prompt = ""
        if known_terms:
            memory_prompt = "请参考以下已知隐私术语，以保持抽取一致性：\n"
            if known_terms:
                memory_prompt += f"【术语示例】{', '.join(known_terms)}\n"

        prompt = (
                memory_prompt
                + guide
                + "\n\n句子列表：\n"
                + json.dumps(indexed_batch, ensure_ascii=False, separators=(",", ":"))

                + "\n\n【正确示例】\n"
                + '输入: [{"idx": 0, "text": "我们会收集您的手机号用于账号注册"}, '
                + '{"idx": 1, "text": "当您使用支付功能时，我们收集您的支付账号信息"}]\n'
                + '输出: {"0": [{"data_type": "手机号", "purpose": "账号注册", "recipients": ["开发者"], "confidence": 0.95}], '
                + '"1": [{"data_type": "支付账号信息", "purpose": "支付", "recipients": ["开发者"], "confidence": 0.9}]}\n\n'

                + "【反例 - 必须输出[]】\n"
                + '1. "我们不会向第三方共享您的个人信息" → []\n'
                + '2. "但以下情况除外：(1)用户本人同意" → []\n'
                + '3. "除非法律法规要求，我们不会披露您的信息" → []\n'
                + '4. "如果您提供以下额外信息，将有助于我们提供更好的服务" → []\n'
                + '5. "若您不提供这类信息，您可能无法正常使用我们的服务" → []\n'
                + '6. "您可以随时撤回授权、删除个人信息" → []\n'
                + '7. "您可以选择是否提供位置信息" → []\n'
                + '8. "我们收集您的信息用于改善服务，但如果您拒绝提供，不会影响基本功能" → []\n'
                + '9. "个人信息是指以电子或其他方式记录的能够识别特定自然人的信息" → []\n'
                + '10. "与国家安全、公共安全直接相关的，我们无需征得您的同意" → []\n\n'

                + "【正例 - 应该提取】\n"
                + '✓ "当您注册时，我们会收集您的手机号码" → 提取（主干：我们会收集）\n'
                + '✓ "为了给您提供服务，我们使用您的位置信息" → 提取（主干：我们使用）\n'
                + '✓ "在支付过程中，我们可能收集您的支付信息" → 提取（主干：我们收集，confidence降低因为有"可能"）\n'
        )
        try:
            out = self.llm.chat(
                prompt=prompt,
                app_id=app_id,
                task_type="policy_extractor"
            )
            # print(out)
        except Exception as e:
            raise RuntimeError(f"[LLMExtractor] LLM 调用失败: {e}")

        if not out or not isinstance(out, str):
            raise ValueError("[LLMExtractor] LLM 返回为空或类型非法。")
        
        return out, prompt

    def _parse_and_validate_output(self, raw: str, expected_indices: List[int], app_id: str) -> List[List[Dict[str, Any]]]:
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                # Fallback to json_repair
                self.logger.warning(f"[{app_id}] 标准 JSON 解析失败，尝试使用 json_repair 修复...")
                parsed = json_repair.loads(raw)
                self.logger.info(f"[{app_id}] json_repair 修复成功。")
            except Exception as e:
                 raise ValueError(
                    f"[LLMExtractor] LLM 输出无法解析(json & json_repair)。原文片段: {raw[:200]}... 错误: {e}"
                )
        # 统一转成 {idx: list[dict]}
        result_map: Dict[int, List[Dict[str, Any]]] = {}

        if isinstance(parsed, dict):
            for key, value in parsed.items():
                try:
                    idx = int(key)
                except Exception:
                    self.logger.warning(
                        f"[LLMExtractor] LLM 输出包含无法识别的键 '{key}'，已忽略。"
                    )
                    continue
                if isinstance(value, list):
                    result_map[idx] = value
                elif isinstance(value, dict):
                    result_map[idx] = [value]
                elif value is None:
                    result_map[idx] = []
                else:
                    self.logger.warning(
                        f"[LLMExtractor] 键 '{key}' 的值类型异常({type(value)}), 已视为缺失。"
                    )
                    result_map[idx] = []
        elif isinstance(parsed, list):
            # 兼容旧格式（list[list] 或 list[dict]）
            normalized: List[List[Dict[str, Any]]] = []
            if parsed and all(isinstance(e, dict) for e in parsed):
                normalized = [parsed]
            else:
                normalized = parsed
            if len(normalized) != len(expected_indices):
                self.logger.warning(
                    f"[LLMExtractor] 输出长度 {len(normalized)} 与期望 {len(expected_indices)} 不符，"
                    "会按顺序补齐/截断。"
                )
            for idx, value in zip(expected_indices, normalized):
                if isinstance(value, list):
                    result_map[idx] = value
                elif value is None:
                    result_map[idx] = []
                else:
                    result_map[idx] = [value] if isinstance(value, dict) else []
            # 如果 normalized 比期望少，多出来的 idx 自动继续补 []
        else:
            raise TypeError(
                f"[LLMExtractor] 解析结果类型不合法: {type(parsed)}"
            )

        triples_list: List[List[Dict[str, Any]]] = []
        for idx in expected_indices:
            triples_list.append(result_map.get(idx, []))

        # 字段校验 - 逐句验证，遇到非法时只清空该句而不中断整体
        for idx, elem in enumerate(triples_list):
            if not isinstance(elem, list):
                self.logger.warning(
                    f"[LLMExtractor] 第 {idx} 个元素不是数组(类型: {type(elem)})，已清空该句结果。"
                )
                triples_list[idx] = []
                continue

            if not elem:
                continue

            # 用于标记当前句子是否有非法三元组
            has_invalid = False
            for j, triple in enumerate(elem):
                try:
                    if not isinstance(triple, dict):
                        self.logger.warning(
                            f"[LLMExtractor] 第 {idx} 句的第 {j} 个三元组不是对象，已清空该句结果。"
                        )
                        has_invalid = True
                        break

                    # 检查必需字段
                    for req in ("data_type", "purpose", "recipients", "confidence"):
                        if req not in triple:
                            self.logger.warning(
                                f"[LLMExtractor] 第 {idx} 句的第 {j} 个三元组缺少字段 {req}，已清空该句结果。"
                            )
                            has_invalid = True
                            break

                    if has_invalid:
                        break

                    # 验证 data_type
                    if not isinstance(triple["data_type"], str) or not triple["data_type"].strip():
                        self.logger.warning(
                            f"[LLMExtractor] 第 {idx} 句的第 {j} 个三元组 data_type 非法，已清空该句结果。"
                        )
                        has_invalid = True
                        break

                    # 验证 purpose
                    if not isinstance(triple["purpose"], str) or not triple["purpose"].strip():
                        self.logger.warning(
                            f"[LLMExtractor] 第 {idx} 句的第 {j} 个三元组 purpose 非法，已清空该句结果。"
                        )
                        has_invalid = True
                        break

                    # 验证 recipients
                    if not isinstance(triple["recipients"], list) or not all(
                        isinstance(r, str) for r in triple["recipients"]
                    ):
                        self.logger.warning(
                            f"[LLMExtractor] 第 {idx} 句的第 {j} 个三元组 recipients 非法，已清空该句结果。"
                        )
                        has_invalid = True
                        break

                    # 验证 confidence
                    try:
                        conf = float(triple["confidence"])
                    except Exception:
                        self.logger.warning(
                            f"[LLMExtractor] 第 {idx} 句的第 {j} 个三元组 confidence 不是数值，已清空该句结果。"
                        )
                        has_invalid = True
                        break

                    if not (0.0 <= conf <= 1.0):
                        self.logger.warning(
                            f"[LLMExtractor] 第 {idx} 句的第 {j} 个三元组 confidence 超出范围({conf})，已清空该句结果。"
                        )
                        has_invalid = True
                        break

                except Exception as e:
                    self.logger.warning(
                        f"[LLMExtractor] 第 {idx} 句的第 {j} 个三元组验证时出现异常: {e}，已清空该句结果。"
                    )
                    has_invalid = True
                    break

            # 如果该句有非法三元组，清空整句结果
            if has_invalid:
                triples_list[idx] = []

        non_empty = any(elem for elem in triples_list)
        if not non_empty:
            self.logger.warning(
                f"[LLMExtractor] 输出结构合法，但在 {len(expected_indices)} 个句子中未检测到任何隐私项。"
            )
        return triples_list

    def _triple_to_item(self, t: Dict[str, Any], sentence: str, sentence_id: int):
        data_type = t["data_type"].strip()
        purpose = t["purpose"].strip()
        recipients = [r.strip() for r in t["recipients"] if isinstance(r, str) and r.strip()]
        confidence = float(t["confidence"])

        item = PrivacyItem(
            data_type=data_type,
            purpose=purpose,
            recipients=recipients,
            confidence=confidence,
            source="llm",
            evidence_text=sentence,
            sentence_id=sentence_id
        )
        return item
    
    def _split_merged_triples(self, triples_per_sentence: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        result = []
        separators = ["、", ",", "，", "和", "与", "及", "或", "以及"]
        for sentence_triples in triples_per_sentence:
            expanded_triples = []
            for triple in sentence_triples:
                data_type = triple["data_type"]
                purpose = triple["purpose"]
                recipients = triple["recipients"]
                confidence = triple["confidence"]

                data_types = self._split_data_type(data_type, separators)
                if len(data_types) > 1:
                    self.logger.info(f"拆分'{data_type}'为{data_types}")
                    for dt in data_types:
                        expanded_triples.append({
                            "data_type": dt,
                            "purpose": purpose,
                            "recipients": recipients,
                            "confidence": confidence
                        })
                else:
                    expanded_triples.append(triple)
            result.append(expanded_triples)
        return result

    def _split_data_type(self, data_type: str, separators: List[str]) -> List[str]:
        has_parenthesis = bool(re.search(r'[（(].*?[）)]', data_type))
        if has_parenthesis:
            self.logger.debug(f"视括号为详细举例，不进行拆分: {data_type}")
            return [data_type.strip()]
        parts = [data_type]
        for sep in separators:
            new_parts = []
            for part in parts:
                if sep in part:
                    split_parts = [p.strip() for p in part.split(sep)]
                    new_parts.extend(split_parts)
                else:
                    new_parts.append(part)
            parts = new_parts
        result = [part.strip() for part in parts if part.strip()]
        if not result:
            return [data_type.strip()]
        if len(result) > 1:
            self.logger.info(f"拆分 '{data_type}' → {result}")
        return result