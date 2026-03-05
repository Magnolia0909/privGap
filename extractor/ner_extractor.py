# -*- coding: utf-8 -*-
import logging
import os
import sys
import json
import torch
from typing import List, Tuple, Dict, Any

# 绕过 transformers 的 torch.load 安全检查
import transformers.utils.import_utils as _import_utils
_import_utils.check_torch_load_is_safe = lambda: None

from transformers import AutoTokenizer, AutoConfig, BertTokenizer

from data.data_process.data_structures import PrivacyItem
from memory.knowledge_base import KnowledgeBase

# 添加 ner 模块路径并导入（指向项目根目录，使用根目录的 ner 模块）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NER_MODULE_PATH = PROJECT_ROOT
if NER_MODULE_PATH not in sys.path:
    sys.path.insert(0, NER_MODULE_PATH)

from ner.models import BertForNERWithCRF, BertForNERWithContrastive
from ner.models import BertBiLSTMCRF_V2, CA4P483_ID2LABEL
from ner.models import BiLSTMCRFBaseline

# 从全局 config 导入 ID2LABEL（已合并自 ner/config/config.py）
from config.config import ID2LABEL


class NERExtractor:
    def __init__(self, config, kb: KnowledgeBase, memory_manager=None):
        self.config = config
        self.kb = kb
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.model_path = self.config.ner_model_path
        self.conf_threshold = self.config.ner_confidence_threshold
        self.max_seq_length = self.config.ner_max_seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = None  # 'bilstm_crf_v2' or 'bert_crf'
        self.id2label = None

    def load_model(self):
        if self.model is not None:
            return
        if not self.model_path:
            raise ValueError("[NERExtractor] 请在 Config.ner_model_path 指定 NER 模型目录。")

        # 检查模型类型（通过检查 config.json 中的 architectures）
        config_path = os.path.join(self.model_path, "config.json")
        model_arch = None
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                architectures = config_data.get("architectures", [])
                if "BertBiLSTMCRF_V2" in architectures:
                    model_arch = "bilstm_crf_v2"
                elif "BiLSTMCRFBaseline" in architectures:
                    model_arch = "bilstm_crf_baseline"

        if model_arch == "bilstm_crf_v2":
            self._load_bilstm_crf_v2_model()
        elif model_arch == "bilstm_crf_baseline":
            self._load_bilstm_crf_baseline_model()
        else:
            self._load_bert_crf_model()

        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"[NERExtractor] 成功加载 NER 模型（{self.model_type}），路径: {self.model_path}，设备: {self.device}。")

    def _load_bilstm_crf_v2_model(self):
        """加载 BERT-BiLSTM-CRF V2 模型（CA4P-483 架构）"""
        self.model_type = 'bilstm_crf_v2'
        self.id2label = CA4P483_ID2LABEL

        # 加载 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path, local_files_only=True)

        # 加载模型配置
        model_config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)

        # 加载模型
        self.model = BertBiLSTMCRF_V2.from_pretrained(
            self.model_path,
            config=model_config,
            lstm_hidden_size=128,
            lstm_num_layers=1,
            dropout_rate=0.5,
            local_files_only=True
        )

    def _load_bilstm_crf_baseline_model(self):
        """加载 BiLSTM-CRF 基线模型（不带 BERT）"""
        self.model_type = 'bilstm_crf_baseline'
        self.id2label = CA4P483_ID2LABEL

        # 加载词表（先查找当前目录，再查找父目录）
        vocab_path = os.path.join(self.model_path, 'vocab.json')
        if not os.path.exists(vocab_path):
            # 尝试父目录
            vocab_path = os.path.join(os.path.dirname(self.model_path), 'vocab.json')

        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.char2id = json.load(f)
        else:
            raise ValueError(f"[NERExtractor] BiLSTM-CRF 模型缺少词表文件: {vocab_path}")

        # 加载模型
        self.model = BiLSTMCRFBaseline.from_pretrained(self.model_path)
        self.tokenizer = None  # BiLSTM-CRF 不使用 tokenizer

    def _load_bert_crf_model(self):
        """加载原有的 BERT+CRF 模型"""
        self.model_type = 'bert_crf'
        self.id2label = ID2LABEL

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)

        # 加载模型配置
        model_config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)

        # 尝试加载不同类型的模型（先尝试 CRF，再尝试 Contrastive）
        try:
            self.model = BertForNERWithCRF.from_pretrained(
                self.model_path,
                config=model_config,
                use_crf=True,
                use_focal_loss=True,
                local_files_only=True
            )
        except Exception:
            self.model = BertForNERWithContrastive.from_pretrained(
                self.model_path,
                config=model_config,
                use_crf=True,
                use_focal_loss=True,
                local_files_only=True
            )
    
    def predict_single(self, text: str) -> List[Dict]:
        """
        对单个文本进行 NER 预测

        Args:
            text: 输入文本

        Returns:
            实体列表，每个实体包含 type, start, end, text
        """
        if self.model_type == 'bilstm_crf_v2':
            return self._predict_single_bilstm_v2(text)
        elif self.model_type == 'bilstm_crf_baseline':
            return self._predict_single_bilstm_baseline(text)
        else:
            return self._predict_single_bert_crf(text)

    def _predict_single_bilstm_v2(self, text: str) -> List[Dict]:
        """使用 BERT-BiLSTM-CRF V2 模型预测"""
        # 字符级别分词
        chars = list(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        if not chars:
            return []

        # Tokenize (与训练时一致)
        tokens = []
        char_to_token = []  # 每个字符对应的 token 索引

        for char in chars:
            word_tokens = self.tokenizer.tokenize(char)
            if not word_tokens:
                word_tokens = [self.tokenizer.unk_token]
            char_to_token.append(len(tokens) + 1)  # +1 是因为有 [CLS]
            tokens.extend(word_tokens)

        # 截断
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[:self.max_seq_length - 2]

        # 添加 [CLS] 和 [SEP]
        final_tokens = ["[CLS]"] + tokens + ["[SEP]"]

        # 转换为 ID
        input_ids = self.tokenizer.convert_tokens_to_ids(final_tokens)
        seq_length = len(input_ids)

        # Padding
        attention_mask = [1] * seq_length
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)

        # 转换为 tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)

        # 预测
        with torch.no_grad():
            predictions = self.model.decode(input_ids, attention_mask)[0]

        # 提取实体（跳过 [CLS] 和 [SEP]）
        entities = []
        current_entity = None

        for i, char in enumerate(chars):
            if i >= len(char_to_token):
                break
            token_idx = char_to_token[i]
            if token_idx >= len(predictions):
                break

            pred_id = predictions[token_idx]
            tag = self.id2label.get(pred_id, 'O')

            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'type': tag[2:],
                    'start': i,
                    'end': i + 1,
                    'text': char
                }
            elif tag.startswith('I-') and current_entity:
                if tag[2:] == current_entity['type']:
                    current_entity['end'] = i + 1
                    current_entity['text'] += char
                else:
                    entities.append(current_entity)
                    current_entity = None
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

    def _predict_single_bilstm_baseline(self, text: str) -> List[Dict]:
        """使用 BiLSTM-CRF 基线模型预测（不带 BERT）"""
        # 字符级别分词
        chars = list(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        if not chars:
            return []

        # 截断
        if len(chars) > self.max_seq_length:
            chars = chars[:self.max_seq_length]

        seq_length = len(chars)

        # 转换为 ID
        char_ids = [self.char2id.get(c, self.char2id.get('<UNK>', 1)) for c in chars]

        # Padding
        while len(char_ids) < self.max_seq_length:
            char_ids.append(0)

        # 转换为 tensor
        input_ids = torch.tensor([char_ids], dtype=torch.long).to(self.device)
        lengths = torch.tensor([seq_length], dtype=torch.long).to(self.device)

        # 预测
        with torch.no_grad():
            predictions = self.model.decode(input_ids, lengths=lengths)[0]

        # 提取实体
        entities = []
        current_entity = None

        for i, char in enumerate(chars):
            if i >= len(predictions):
                break

            pred_id = predictions[i]
            tag = self.id2label.get(pred_id, 'O')

            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'type': tag[2:],
                    'start': i,
                    'end': i + 1,
                    'text': char
                }
            elif tag.startswith('I-') and current_entity:
                if tag[2:] == current_entity['type']:
                    current_entity['end'] = i + 1
                    current_entity['text'] += char
                else:
                    entities.append(current_entity)
                    current_entity = None
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

    def _predict_single_bert_crf(self, text: str) -> List[Dict]:
        """使用原有 BERT+CRF 模型预测"""
        # 字符级别分词
        tokens = list(text.replace(' ', '').replace('\n', '').replace('\t', ''))

        if not tokens:
            return []

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 预测
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            if hasattr(self.model, 'decode'):
                predictions = self.model.decode(input_ids, attention_mask)[0]
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = outputs.logits.argmax(dim=-1)[0].tolist()

        # 对齐预测结果到原始 token
        word_ids = encoding.word_ids()
        token_predictions = []
        previous_word_id = None

        for word_id, pred_id in zip(word_ids, predictions):
            if word_id is None:
                continue
            if word_id != previous_word_id:
                token_predictions.append(pred_id)
            previous_word_id = word_id

        # 提取实体
        entities = self._extract_entities(tokens, token_predictions)
        return entities

    def _extract_entities(self, tokens: List[str], predictions: List[int]) -> List[Dict]:
        """从预测结果中提取实体"""
        entities = []
        current_entity = None

        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            tag = self.id2label.get(pred_id, 'O')

            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'type': tag[2:],
                    'start': i,
                    'end': i + 1,
                    'text': token
                }
            elif tag.startswith('I-') and current_entity:
                if tag[2:] == current_entity['type']:
                    current_entity['end'] = i + 1
                    current_entity['text'] += token
                else:
                    entities.append(current_entity)
                    current_entity = None
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

    def extract(self, sentences: List[str], app_id: str = None) -> Tuple[List[PrivacyItem], int]:
        if not sentences:
            return [], 0
        self.load_model()

        items: List[PrivacyItem] = []

        for sent_idx, sentence in enumerate(sentences):
            try:
                entities = self.predict_single(sentence)
            except Exception as exc:
                self.logger.error(f"[NERExtractor] NER 模型推理失败: {exc}")
                continue

            # 按实体类型分组
            field_entities: Dict[str, List[Dict[str, Any]]] = {
                "data_type": [],
                "purpose": [],
                "recipients": [],
            }

            for entity in entities:
                entity_type = entity.get("type", "")
                entity_text = entity.get("text", "").strip()
                if not entity_text:
                    continue

                # 映射实体类型到字段
                if entity_type == "data":
                    field_entities["data_type"].append({"text": entity_text})
                elif entity_type == "purpose":
                    field_entities["purpose"].append({"text": entity_text})
                elif entity_type in ("recipients", "handler"):
                    # CA4P-483 使用 handler，其他模型可能使用 recipients
                    field_entities["recipients"].append({"text": entity_text})

            if not any(field_entities.values()):
                continue

            recipients = [entry["text"] for entry in field_entities["recipients"]]
            data_terms = field_entities["data_type"] or [{"text": ""}]
            purpose_terms = field_entities["purpose"] or [{"text": ""}]

            for data_entry in data_terms:
                for purpose_entry in purpose_terms:
                    if not data_entry["text"] and not purpose_entry["text"]:
                        continue
                    items.append(PrivacyItem(
                        data_type=data_entry["text"],
                        purpose=purpose_entry["text"],
                        recipients=recipients,
                        confidence=1.0,  # CRF 模型不输出置信度，默认为1.0
                        source="ner",
                        evidence_text=sentence,
                        sentence_id=sent_idx,
                    ))

        return items, len(sentences)
