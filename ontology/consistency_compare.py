# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from config.config import Config
from memory.knowledge_base import KnowledgeBase
from main import evaluate_datatype_consistency_onto


def _resolve_items_dir(root: str, kind: str, prefer: Optional[list[str]] = None) -> str:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{kind} 目录不存在: {root}")

    def has_items(path: str) -> bool:
        return any(name.endswith("_privacy_items.json") for name in os.listdir(path))

    if has_items(root):
        return root

    prefer = prefer or []
    for sub in prefer:
        cand = os.path.join(root, sub)
        if os.path.isdir(cand) and has_items(cand):
            return cand

    # 如果只有一个子目录，自动选择
    subdirs = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    if len(subdirs) == 1 and has_items(subdirs[0]):
        return subdirs[0]

    raise FileNotFoundError(
        f"{kind} 目录下未找到 *_privacy_items.json: {root}"
    )


def _resolve_normalized_dir(root: str, kind: str, suffix: str) -> str:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{kind} 目录不存在: {root}")
    if any(name.endswith(suffix) for name in os.listdir(root)):
        return root
    raise FileNotFoundError(f"{kind} 目录下未找到 *{suffix}: {root}")


def _materialize_privacy_items_dir(src_dir: str, suffix: str, kind: str) -> str:
    temp_dir = tempfile.mkdtemp(prefix=f"consistency_{kind}_")
    for fname in os.listdir(src_dir):
        if not fname.endswith(suffix):
            continue
        app_id = fname[: -len(suffix)]
        src_path = os.path.join(src_dir, fname)
        with open(src_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        dst_path = os.path.join(temp_dir, f"{app_id}_privacy_items.json")
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    return temp_dir


def _collect_permissions(dir_path: str, kind: str, limit: int = 20) -> list[str]:
    perms = set()
    missing_match = 0
    total_items = 0
    for fname in os.listdir(dir_path):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(dir_path, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            total_items += 1
            match = item.get("ontology_match")
            if isinstance(match, dict) and match.get("hierarchy_path"):
                perms.add(match["hierarchy_path"])
                continue
            missing_match += 1
            dt = (item.get("data_type", "") or "").strip()
            if dt:
                perms.add(dt)
    sample = sorted(perms)[:limit]
    print(
        f"[{kind}] 权限总数: {len(perms)}; 示例: {sample}; "
        f"未映射条目: {missing_match}/{total_items}"
    )
    return sample


def _build_cfg(output_root: str, llm_backend: str) -> Config:
    cfg = Config()
    cfg.OUTPUT_DIR = output_root
    cfg.FILTER_DIR = os.path.join(cfg.OUTPUT_DIR, "filter")
    cfg.test_result_dir = os.path.join(cfg.OUTPUT_DIR, "extractions")
    cfg.llm_backend = llm_backend
    return cfg


def run_compare(
    run_tag: str,
    guide_dir: str,
    policy_dir: str,
    output_root: str,
    llm_backend: str,
):
    print(f"[{run_tag}] guide_dir={guide_dir}")
    print(f"[{run_tag}] policy_dir={policy_dir}")
    _collect_permissions(guide_dir, f"{run_tag} guide")
    _collect_permissions(policy_dir, f"{run_tag} policy")
    cfg = _build_cfg(output_root, llm_backend)
    kb = KnowledgeBase(cfg)
    print(f"[{run_tag}] 开始一致性评估...")
    return evaluate_datatype_consistency_onto(cfg, kb, guide_dir=guide_dir, policy_dir=policy_dir)


def main():
    parser = argparse.ArgumentParser(description="Compare consistency for base vs onto using main.py logic.")
    parser.add_argument(
        "--llm-backend",
        default="claude",
        help="LLM backend used by ontology classifier (default: claude).",
    )
    parser.add_argument(
        "--guide-base",
        default="/home/sxx/experiment/privGap/results/guide/extractions",
        help="Base guideline extraction root directory.",
    )
    parser.add_argument(
        "--policy-base",
        default="/home/sxx/experiment/privGap/results/claude_base/extractions",
        help="Base policy extraction root directory.",
    )
    parser.add_argument(
        "--guide-onto",
        default="/home/sxx/experiment/privGap/results/claude_onto/consistency/guide_normalized",
        help="Ontology guideline normalized directory.",
    )
    parser.add_argument(
        "--policy-onto",
        default="/home/sxx/experiment/privGap/results/claude_onto/consistency/policy_normalized",
        help="Ontology policy normalized directory.",
    )
    parser.add_argument(
        "--output-base",
        default="/home/sxx/experiment/privGap/results/claude_base_consistency",
        help="Output root for base comparison results.",
    )
    parser.add_argument(
        "--output-onto",
        default="/home/sxx/experiment/privGap/results/claude_onto_consistency",
        help="Output root for ontology comparison results.",
    )
    args = parser.parse_args()

    guide_base_dir = _resolve_items_dir(
        args.guide_base,
        "guide(base)",
        prefer=["guide"],
    )
    policy_base_dir = _resolve_items_dir(
        args.policy_base,
        "policy(base)",
        prefer=["policy_llm_ner", "policy_llm", "policy"],
    )
    guide_onto_dir = _resolve_items_dir(
        args.guide_onto,
        "guide(onto)",
    ) if os.path.isdir(args.guide_onto) and any(
        name.endswith("_privacy_items.json") for name in os.listdir(args.guide_onto)
    ) else _resolve_normalized_dir(
        args.guide_onto,
        "guide(onto)",
        "_guide_normalized.json",
    )
    policy_onto_dir = _resolve_items_dir(
        args.policy_onto,
        "policy(onto)",
    ) if os.path.isdir(args.policy_onto) and any(
        name.endswith("_privacy_items.json") for name in os.listdir(args.policy_onto)
    ) else _resolve_normalized_dir(
        args.policy_onto,
        "policy(onto)",
        "_policy_normalized.json",
    )

    if not any(name.endswith("_privacy_items.json") for name in os.listdir(guide_onto_dir)):
        guide_onto_dir = _materialize_privacy_items_dir(
            guide_onto_dir, "_guide_normalized.json", "guide_onto"
        )
    if not any(name.endswith("_privacy_items.json") for name in os.listdir(policy_onto_dir)):
        policy_onto_dir = _materialize_privacy_items_dir(
            policy_onto_dir, "_policy_normalized.json", "policy_onto"
        )

    run_compare(
        run_tag="claude_base",
        guide_dir=guide_base_dir,
        policy_dir=policy_base_dir,
        output_root=args.output_base,
        llm_backend=args.llm_backend,
    )
    run_compare(
        run_tag="claude_onto",
        guide_dir=guide_onto_dir,
        policy_dir=policy_onto_dir,
        output_root=args.output_onto,
        llm_backend=args.llm_backend,
    )


if __name__ == "__main__":
    main()
