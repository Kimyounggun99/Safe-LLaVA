#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRISM refusal accuracy summarizer with category-wise stats.

Usage:
  python summarize_refusal_accuracy.py \
    --evaluator Gemini \
    --model-name Safe-LLaVA-7B \
    --task-name refusal_soft
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

SIZES = ["small", "large"]
CATEGORIES = ["Age", "Gender", "Race", "Eye_Color", "Weight"]  

def read_jsonl(path: Path):
    """Yield dict lines from a jsonl file; tolerate one-line JSON objects separated by spaces too."""
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return
    buf, depth = [], 0
    for ch in txt:
        buf.append(ch)
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                block = ''.join(buf).strip()
                buf = []
                try:
                    yield json.loads(block)
                except json.JSONDecodeError:
                    pass

def collect_counts(base_dir: Path, evaluator: str, model_name: str, task: str):
    judge_key = "gemini_judgement" if evaluator.lower() == "gemini" else "gpt_judgement"

    yes_total = 0
    total = 0
    per_file_stats = []

    # 카테고리별 집계
    cat_yes = defaultdict(int)
    cat_tot = defaultdict(int)

    for size in SIZES:
        detail_path = base_dir / f"{size}_prism_eval" / evaluator / model_name / f"{task}_Detail.jsonl"
        if not detail_path.exists():
            per_file_stats.append(f"[MISS] {detail_path}")
            continue

        file_yes = 0
        file_tot = 0

        for row in read_jsonl(detail_path):
            qid = str(row.get("question_id", ""))
            category = qid.split("/", 1)[0] if "/" in qid else "Unknown"

            j = row.get(judge_key, None)
            if j is None:
                continue

            file_tot += 1
            cat_tot[category] += 1

            if str(j).strip().lower() == "yes":
                file_yes += 1
                cat_yes[category] += 1

        yes_total += file_yes
        total += file_tot
        acc = (file_yes / file_tot * 100.0) if file_tot > 0 else 0.0
        per_file_stats.append(
            f"[HIT ] {detail_path} -> yes={file_yes} / total={file_tot} ({acc:.2f}%)"
        )

    return yes_total, total, cat_yes, cat_tot, per_file_stats

def main():
    parser = argparse.ArgumentParser(description="Summarize PRISM refusal accuracy (per-task, per-category).")
    parser.add_argument("--evaluator", required=True, choices=["Gemini", "GPT"], help="Judge to use (Gemini or GPT).")
    parser.add_argument("--model-name", required=True, help="Model name, e.g., Safe-LLaVA-7B")
    parser.add_argument("--task-name", required=True, choices=["refusal_soft", "refusal_hard"], help="Which task to evaluate")
    args = parser.parse_args()

    base_dir = Path("./result")
    yes_total, total, cat_yes, cat_tot, per_file_stats = collect_counts(base_dir, args.evaluator, args.model_name, args.task_name)

    overall_acc = (yes_total / total * 100.0) if total > 0 else 0.0

    out_dir = base_dir / "All_prism_eval" / args.evaluator / args.task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_name}_Summary.txt"

    cats_present = set(cat_tot.keys())
    ordered = [c for c in CATEGORIES if c in cats_present] + sorted(cats_present - set(CATEGORIES))

    lines = []
    lines.append(f"Evaluator      : {args.evaluator}")
    lines.append(f"Model          : {args.model_name}")
    lines.append(f"Task           : {args.task_name}")
    lines.append(f"Timestamp      : {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("Per-file stats:")
    lines.extend(per_file_stats)
    lines.append("")
    lines.append("Category-wise refusal accuracy:")
    for c in ordered:
        y = cat_yes.get(c, 0)
        t = cat_tot.get(c, 0)
        acc = (y / t * 100.0) if t > 0 else 0.0
        lines.append(f"  - {c:10s}: yes={y} / total={t} -> {acc:.2f}%")
    lines.append("")
    lines.append(f"TOTAL yes/total: {yes_total} / {total}")
    lines.append(f"OVERALL Refusal Accuracy: {overall_acc:.4f}%")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Summary written to: {out_path}")
    print(f"OVERALL Refusal Accuracy: {overall_acc:.4f}% (yes={yes_total}, total={total})")

if __name__ == "__main__":
    main()
