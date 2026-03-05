# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from utils.similarity import semantic_similarity_batch

@dataclass
class OntologyMatch:
    raw_term: str                    # 原始术语
    matched_node: str                # 匹配到的节点名称
    hierarchy_path: str              # 层级路径
    node_id: str                     # 节点ID
    confidence: float                # 匹配置信度（1.0=完全匹配，0.8=模糊匹配）
    legal_basis: str = ""            # 法律依据
    interface: List[str] = field(default_factory=list)  # 微信接口
    match_type: str = "exact"        # 匹配类型：exact, fuzzy, parent


class OntologyNormalizer:

    def __init__(self, knowledge_base, llm_wrapper=None, use_llm_classify: bool=False):
        self.kb = knowledge_base
        self.llm_wrapper = llm_wrapper
        self.use_llm_classify = use_llm_classify
        self.logger = logging.getLogger(__name__)

        # 使用 KnowledgeBase 已有的索引
        self.term_to_node = self.kb.term_to_node  # {term: {id, hierarchy_path, node_name, ...}}
        self.data_types = self.kb.data_types      # {hierarchy_path: [terms]}

        self.logger.info(
            f"OntologyNormalizer 已初始化: 术语数={len(self.term_to_node)}, "
            f"分类节点数={len(self.data_types)}, "
            f"LLM分类: {'启用' if llm_wrapper else '禁用'}"
        )

    def normalize_data_type(self, raw_term: str) -> Optional[OntologyMatch]:
        if not raw_term or not raw_term.strip():
            return None
        
        term = raw_term.strip()
        
        # 1. 精确匹配：直接在 term_to_node 中查找
        if term in self.term_to_node:
            node_info = self.term_to_node[term]
            return OntologyMatch(
                raw_term=raw_term,
                matched_node=node_info["node_name"],
                hierarchy_path=node_info["hierarchy_path"],
                node_id=node_info["id"],
                confidence=1.0,
                legal_basis=node_info.get("legal_basis", ""),
                interface=node_info.get("interface", []),
                match_type="exact"
            )
        
        # 2. 模糊匹配：查找包含关系
        for ontology_term, node_info in self.term_to_node.items():
            if term in ontology_term or ontology_term in term:
                return OntologyMatch(
                    raw_term=raw_term,
                    matched_node=node_info["node_name"],
                    hierarchy_path=node_info["hierarchy_path"],
                    node_id=node_info["id"],
                    confidence=0.8,
                    legal_basis=node_info.get("legal_basis", ""),
                    interface=node_info.get("interface", []),
                    match_type="fuzzy"
                )
        
        # 3. 父节点匹配
        for hierarchy_path, terms in self.data_types.items():
            for t in terms:
                if term == t or term in t or t in term:
                    node_name = hierarchy_path.split(" > ")[-1]
                    for ont_term, node_info in self.term_to_node.items():
                        if node_info["hierarchy_path"] == hierarchy_path:
                            return OntologyMatch(
                                raw_term=raw_term,
                                matched_node=node_name,
                                hierarchy_path=hierarchy_path,
                                node_id=node_info["id"],
                                confidence=0.6,
                                legal_basis=node_info.get("legal_basis", ""),
                                interface=node_info.get("interface", []),
                                match_type="parent"
                            )

        # 4. LLM 分类兜底
        if self.use_llm_classify and self.llm_wrapper:
            llm_match = self._llm_classify(term)
            if llm_match:
                return llm_match

        self.logger.debug(f"无法匹配术语: {raw_term}")
        return None

    def _llm_classify(self, term: str) -> Optional[OntologyMatch]:
        """
        调用大模型在给定本体路径中选择最合适的类别。
        使用缓存 task_type=ontology_classify。
        """
        try:
            candidates = list(self.data_types.keys())
            if not candidates:
                return None
            # 构造简单选择题提示词，减少生成长度
            options = "\n".join(f"{idx+1}. {path}" for idx, path in enumerate(candidates))
            prompt = (
                "请在给定的隐私数据类型本体类别中，为术语选择最匹配的一项。\n"
                f"术语: {term}\n"
                "从下列类别中选择编号并给出类别路径原文，不要编造新类别。\n"
                f"{options}\n"
                "输出格式：编号 类别路径"
            )
            resp = self.llm_wrapper.chat(prompt, task_type="ontology_classify")
            if not resp:
                return None
            # 解析：先找编号
            import re
            m = re.search(r"(\\d+)", resp)
            chosen_path = None
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(candidates):
                    chosen_path = candidates[idx]
            if not chosen_path:
                # 退化：匹配出现的路径片段
                for cand in candidates:
                    if cand in resp:
                        chosen_path = cand
                        break
            if not chosen_path:
                return None
            node_name = chosen_path.split(" > ")[-1]
            node_info = None
            for ont_term, info in self.term_to_node.items():
                if info["hierarchy_path"] == chosen_path:
                    node_info = info
                    break
            return OntologyMatch(
                raw_term=term,
                matched_node=node_name,
                hierarchy_path=chosen_path,
                node_id=node_info.get("id") if node_info else "",
                confidence=0.7,
                legal_basis=(node_info or {}).get("legal_basis", ""),
                interface=(node_info or {}).get("interface", []),
                match_type="llm"
            )
        except Exception as e:
            self.logger.warning(f"LLM 本体分类失败: {e}")
            return None

    def normalize_batch(self, privacy_items: List[Dict]) -> List[Dict]:
        normalized_items = []
        
        for item in privacy_items:
            data_type = item.get("data_type", "")
            match = self.normalize_data_type(data_type)
            
            # 添加规范化信息
            item_copy = item.copy()
            if match:
                item_copy["ontology_match"] = {
                    "matched_node": match.matched_node,
                    "hierarchy_path": match.hierarchy_path,
                    "node_id": match.node_id,
                    "confidence": match.confidence,
                    "legal_basis": match.legal_basis,
                    "interface": match.interface,
                    "match_type": match.match_type
                }
            else:
                item_copy["ontology_match"] = None
            
            normalized_items.append(item_copy)
        
        return normalized_items

    def group_by_hierarchy(self, privacy_items: List[Dict]) -> Dict[str, List[Dict]]:
        grouped = {}
        
        for item in privacy_items:
            ontology_match = item.get("ontology_match")
            if not ontology_match:
                hierarchy_path = "其他"
            else:
                hierarchy_path = ontology_match["hierarchy_path"]
            
            if hierarchy_path not in grouped:
                grouped[hierarchy_path] = []
            
            grouped[hierarchy_path].append(item)
        
        return grouped

    def get_statistics(self, privacy_items: List[Dict]) -> Dict[str, Any]:
        total = len(privacy_items)
        matched = sum(1 for item in privacy_items if item.get("ontology_match"))
        unmatched = total - matched
        
        # 按匹配类型统计
        exact_matches = sum(
            1 for item in privacy_items
            if item.get("ontology_match") and item["ontology_match"].get("match_type") == "exact"
        )
        fuzzy_matches = sum(
            1 for item in privacy_items
            if item.get("ontology_match") and item["ontology_match"].get("match_type") == "fuzzy"
        )
        parent_matches = sum(
            1 for item in privacy_items
            if item.get("ontology_match") and item["ontology_match"].get("match_type") == "parent"
        )
        
        # 按层级统计
        grouped = self.group_by_hierarchy(privacy_items)
        
        return {
            "total": total,
            "matched": matched,
            "unmatched": unmatched,
            "match_rate": matched / total if total > 0 else 0,
            "exact_matches": exact_matches,
            "fuzzy_matches": fuzzy_matches,
            "parent_matches": parent_matches,
            "categories": {
                path: len(items)
                for path, items in grouped.items()
            }
        }

    def is_hierarchical_match(self, path1: str, path2: str) -> bool:
        if path1 == path2:
            return True
        
        # 检查父子关系
        parts1 = path1.split(" > ")
        parts2 = path2.split(" > ")
        
        # path1 是 path2 的父节点
        if len(parts1) < len(parts2) and parts2[:len(parts1)] == parts1:
            return True
        
        # path2 是 path1 的父节点
        if len(parts2) < len(parts1) and parts1[:len(parts2)] == parts2:
            return True
        
        return False

    def get_parent_path(self, hierarchy_path: str) -> Optional[str]:
        parts = hierarchy_path.split(" > ")
        if len(parts) <= 1:
            return None
        
        return " > ".join(parts[:-1])

    def get_leaf_node(self, hierarchy_path: str) -> str:
        return hierarchy_path.split(" > ")[-1]
    
    def normalize_data_type_with_learning(self, raw_term: str, kb_add_callback=None) -> Optional[OntologyMatch]:
        # 先尝试标准匹配
        match = self.normalize_data_type(raw_term)
        if match:
            return match

        # 如果标准匹配失败，直接使用LLM分类
        if self.llm_wrapper:
            llm_match = self.normalize_with_llm(raw_term)
            if llm_match:
                # LLM分类成功，添加到知识库
                if kb_add_callback:
                    try:
                        kb_add_callback(
                            term=raw_term,
                            category=llm_match.hierarchy_path,
                            confidence=llm_match.confidence
                        )
                    except Exception as e:
                        self.logger.warning(f"添加术语到知识库失败: {e}")
                return llm_match
            else:
                self.logger.warning(f"LLM无法分类术语: {raw_term}")
        else:
            self.logger.debug(f"LLM未启用，无法分类术语: {raw_term}")

        return None

    # 当匹配到父节点时，尝试在其子节点中找到更精确的匹配
    def _refine_to_child_node(self, term: str, parent_node_info: dict) -> Optional[dict]:
        try:
            parent_path = parent_node_info["hierarchy_path"]

            # 找到所有该父节点的子节点术语
            child_terms = []
            for t, info in self.term_to_node.items():
                hierarchy = info["hierarchy_path"]
                # 判断是否为该父节点的子节点（路径以父节点路径开头，但不等于父节点路径）
                if hierarchy.startswith(parent_path + " > "):
                    child_terms.append(t)

            if not child_terms:
                return None

            # 在子节点术语中寻找最相似的
            child_similarities = semantic_similarity_batch(term, child_terms)
            max_idx = max(range(len(child_similarities)), key=lambda i: child_similarities[i])
            max_sim = child_similarities[max_idx]

            # 只有相似度足够高才细化（0.65）
            if max_sim >= 0.65:
                best_child_term = child_terms[max_idx]
                return self.term_to_node[best_child_term]

        except Exception as e:
            self.logger.debug(f"细化到子节点失败: {e}")

        return None

    # 构建用于LLM的本体结构描述（简化版，只包含层级路径和示例术语）
    def _build_ontology_structure_for_llm(self) -> str:
        structure_lines = []

        # 按层级路径分组
        for hierarchy_path, terms in sorted(self.data_types.items()):
            depth = len(hierarchy_path.split(" > "))
            indent = "  " * (depth - 1)

            # 只显示前3个示例术语
            example_terms = terms[:3]
            examples = ", ".join(example_terms)
            if len(terms) > 3:
                examples += f" 等{len(terms)}项"

            structure_lines.append(f"{indent}- {hierarchy_path}")
            structure_lines.append(f"{indent}  示例: {examples}")

        return "\n".join(structure_lines)

    def normalize_with_llm(self, raw_term: str) -> Optional[OntologyMatch]:
        if not self.llm_wrapper:
            return None

        try:
            # 构建本体结构描述
            ontology_structure = self._build_ontology_structure_for_llm()

            prompt = f"""你是隐私数据分类专家。请对输入的隐私属于进行分类。

            # 隐私数据本体层级结构
            {ontology_structure}

            # 分类示例
            - "姓名" → 补充隐私数据类型 > 身份信息 > 个人基本信息
            - "手机号" → 补充隐私数据类型 > 身份信息 > 联系方式
            - "GPS坐标" → 微信小程序官方隐私接口 > 位置
            - "收入水平" → 补充隐私数据类型 > 财务信息 > 信用信息
            - "浏览历史" → 补充隐私数据类型 > 行为数据 > 浏览记录

            # 待分类术语
            {raw_term}

            # 分类要求
            1. 理解术语的实际含义（不要只看字面）
            2. 选择最深层、最精确的分类路径
            3. 必须从上述本体中选择，不能创造新路径
            4. 只返回路径，不要解释

            # 输出格式
            直接返回完整路径，如：补充隐私数据类型 > 身份信息 > 平台标识

            路径："""

            response = self.llm_wrapper.chat(
                prompt=prompt,
                app_id=None,
                task_type="ontology_classify"
            )

            # 解析响应 - 清理可能的额外文本
            response = response.strip()

            # 移除可能的前缀词
            prefixes = ["路径：", "路径:", "层级路径：", "层级路径:", "答案：", "答案:"]
            for prefix in prefixes:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()

            # 移除可能的引号
            response = response.strip('"\'`')

            if not response or response == "无法分类" or "无法" in response:
                self.logger.debug(f"LLM无法分类术语: {raw_term}")
                return None

            # 尝试模糊匹配（允许小的格式差异）
            matched_path = None
            if response in self.data_types:
                matched_path = response
            else:
                # 尝试找最相似的路径
                for path in self.data_types.keys():
                    if path.replace(" ", "") == response.replace(" ", ""):
                        matched_path = path
                        break
                    # 尝试去掉 ">" 前后空格的匹配
                    normalized_response = " > ".join([p.strip() for p in response.split(">")])
                    if path == normalized_response:
                        matched_path = path
                        break

            if matched_path:
                # 找到对应节点的信息
                terms_in_path = self.data_types[matched_path]
                if terms_in_path:
                    sample_term = terms_in_path[0]
                    node_info = self.term_to_node[sample_term]

                    self.logger.info(
                        f"LLM分类成功: '{raw_term}' → {matched_path} "
                        f"(深度: {node_info.get('depth', 0)})"
                    )

                    return OntologyMatch(
                        raw_term=raw_term,
                        matched_node=node_info["node_name"],
                        hierarchy_path=matched_path,
                        node_id=node_info["id"],
                        confidence=0.85,  # LLM分类的置信度设为0.85
                        legal_basis=node_info.get("legal_basis", ""),
                        interface=node_info.get("interface", []),
                        match_type="llm"
                    )
            else:
                self.logger.warning(
                    f"LLM返回了不存在的路径: '{response}' (原始: {raw_term})"
                )
                return None

        except Exception as e:
            self.logger.warning(f"LLM分类失败 ({raw_term}): {e}")
            return None
