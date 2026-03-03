#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_confidence_from_logits(logits: torch.Tensor, top_k: int = 20) -> float:
    """-mean(log p_topk)."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
    return float((-torch.log(top_probs).mean()).item())


@dataclass
class DynamicThresholds:
    cd_threshold: float
    mask_threshold_lower: float
    mask_threshold_upper: float


class DynamicConfBar:
    """Dynamic thresholds"""

    def __init__(
        self,
        window_size: int = 2048,
        warmup_size: int = 512,
        cd_percentile: float = 3.0,
        mask_percentile_lower: float = 95.0,
        mask_percentile_upper: float = 100.0,
        min_samples: int = 64,
    ):
        self.window_size = int(window_size)
        self.warmup_size = int(warmup_size)
        self.cd_percentile = float(cd_percentile)
        self.mask_percentile_lower = float(mask_percentile_lower)
        self.mask_percentile_upper = float(mask_percentile_upper)
        self.min_samples = int(min_samples)

        self._conf_window: List[float] = []
        self._cd_threshold = None
        self._mask_threshold_lower = None
        self._mask_threshold_upper = None

    def add_confidence(self, conf_value: float) -> None:
        self._conf_window.append(float(conf_value))
        if len(self._conf_window) > self.window_size + 1:
            self._conf_window.pop(0)

    def update_thresholds(self, step: int) -> None:
        if step < self.warmup_size:
            return
        if len(self._conf_window) < self.warmup_size:
            return

        if step < self.window_size:
            confs = self._conf_window[:-1] if len(self._conf_window) > 1 else []
        else:
            confs = self._conf_window[:-1][-self.window_size :] if len(self._conf_window) > 1 else []

        if len(confs) < self.min_samples:
            return

        arr = np.asarray(confs, dtype=np.float64)

        cd_idx = int(len(arr) * self.cd_percentile / 100.0)
        cd_idx = max(0, min(cd_idx, len(arr) - 1))
        self._cd_threshold = float(np.partition(arr, cd_idx)[cd_idx])

        lower_pct = float(self.mask_percentile_lower)
        upper_pct = float(self.mask_percentile_upper)

        lower_idx = int(len(arr) * lower_pct / 100.0)
        lower_idx = max(0, min(lower_idx, len(arr) - 1))
        self._mask_threshold_lower = float(np.partition(arr, lower_idx)[lower_idx])

        if upper_pct >= 100.0:
            self._mask_threshold_upper = float(np.max(arr))
        else:
            upper_idx = int(len(arr) * upper_pct / 100.0)
            upper_idx = max(0, min(upper_idx, len(arr) - 1))
            self._mask_threshold_upper = float(np.partition(arr, upper_idx)[upper_idx])

    def thresholds(self) -> DynamicThresholds:
        return DynamicThresholds(
            cd_threshold=self._cd_threshold if self._cd_threshold is not None else float("inf"),
            mask_threshold_lower=self._mask_threshold_lower if self._mask_threshold_lower is not None else float("inf"),
            mask_threshold_upper=self._mask_threshold_upper if self._mask_threshold_upper is not None else float("inf"),
        )

    def should_apply_cd(self, step: int, conf: float) -> bool:
        if step < self.warmup_size:
            return False
        if self._cd_threshold is None:
            return False
        return float(conf) < float(self._cd_threshold)

    def should_mask(self, step: int, conf: float) -> bool:
        if step < self.warmup_size:
            return False
        if self._mask_threshold_lower is None or self._mask_threshold_upper is None:
            return False
        return float(self._mask_threshold_lower) <= float(conf) <= float(self._mask_threshold_upper)


def _infer_think_token_ids(tokenizer) -> Tuple[int, int]:
    """Get token ids for <think> and </think>."""
    think_start = tokenizer.convert_tokens_to_ids("<think>")
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    if isinstance(think_start, list):
        think_start = think_start[0]
    if isinstance(think_end, list):
        think_end = think_end[0]
    if think_start is None or think_end is None or think_start == tokenizer.unk_token_id or think_end == tokenizer.unk_token_id:
        raise ValueError("tokenizer does not contain <think> or </think>")
    return int(think_start), int(think_end)


def _init_in_think_block(prompt_token_ids: List[int], think_start_id: int, think_end_id: int) -> bool:
    """Init think state from prompt tokens."""
    last_start = -1
    last_end = -1
    for i, tid in enumerate(prompt_token_ids):
        if tid == think_start_id:
            last_start = i
        elif tid == think_end_id:
            last_end = i
    return (last_start != -1) and (last_end < last_start)


@torch.no_grad()
def generate_open_cd(
    model,
    tokenizer,
    prompt_text: str,
    *,
    hc_id: int,
    seed: int = 0,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k_conf: int = 20,
    cd_alpha: float = 0.5,
    use_dynamic_threshold: bool = True,
    window_size: int = 2048,
    warmup_size: int = 512,
    cd_percentile: float = 3.0,
    mask_percentile_lower: float = 95.0,
    mask_percentile_upper: float = 100.0,
) -> Dict[str, Any]:
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_len = int(input_ids.size(1))
    eos_token_id = tokenizer.eos_token_id

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    think_start_id, think_end_id = _infer_think_token_ids(tokenizer)

    main_ids = input_ids.clone()
    past_key_values = None

    cd_ids: List[int] = input_ids[0].tolist()
    past_key_values_cd = None

    in_think_block = _init_in_think_block(cd_ids, think_start_id, think_end_id)

    conf_bar = None
    if use_dynamic_threshold:
        conf_bar = DynamicConfBar(
            window_size=window_size,
            warmup_size=warmup_size,
            cd_percentile=cd_percentile,
            mask_percentile_lower=mask_percentile_lower,
            mask_percentile_upper=mask_percentile_upper,
        )

    cd_steps = 0
    hc_steps = 0

    for step in range(int(max_new_tokens)):
        if past_key_values is None:
            out = model(input_ids=main_ids, use_cache=True)
        else:
            out = model(input_ids=main_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
        logits_original = out.logits[:, -1, :]
        past_key_values = out.past_key_values

        if past_key_values_cd is None:
            out_cd = model(input_ids=torch.tensor([cd_ids], dtype=torch.long, device=device), use_cache=True)
            past_key_values_cd = out_cd.past_key_values
        else:
            last_cd = torch.tensor([[cd_ids[-1]]], dtype=torch.long, device=device)
            out_cd = model(input_ids=last_cd, past_key_values=past_key_values_cd, use_cache=True)
            past_key_values_cd = out_cd.past_key_values
        logits_cd = out_cd.logits[:, -1, :]

        conf_original = compute_confidence_from_logits(logits_original[0], top_k=top_k_conf)
        if conf_bar is not None:
            conf_bar.add_confidence(conf_original)
            conf_bar.update_thresholds(step)
            th = conf_bar.thresholds()
        else:
            th = DynamicThresholds(cd_threshold=float("inf"), mask_threshold_lower=float("inf"), mask_threshold_upper=float("inf"))

        apply_cd = False
        if in_think_block and conf_bar is not None:
            apply_cd = conf_bar.should_apply_cd(step, conf_original)

        if apply_cd:
            cd_logits = (1.0 + float(cd_alpha)) * logits_original - float(cd_alpha) * logits_cd
            conf_used = compute_confidence_from_logits(cd_logits[0], top_k=top_k_conf)
            logits_to_sample = cd_logits
            cd_steps += 1
        else:
            conf_used = conf_original
            logits_to_sample = logits_original

        if temperature and temperature > 0:
            logits_to_sample = logits_to_sample / float(temperature)
        probs = torch.softmax(logits_to_sample, dim=-1)

        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            remove = cum > float(top_p)
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0
            idx_remove = remove.scatter(1, sorted_idx, remove)
            probs = probs.masked_fill(idx_remove, 0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = int(next_token.item())
        main_ids = torch.cat([main_ids, next_token], dim=-1)

        next_in_think = in_think_block
        if next_token_id == think_start_id:
            next_in_think = True
        elif next_token_id == think_end_id:
            next_in_think = False

        if next_in_think and (conf_bar is not None) and conf_bar.should_mask(step, conf_used):
            cd_ids.append(int(hc_id))
            hc_steps += 1
        else:
            cd_ids.append(next_token_id)

        in_think_block = next_in_think

        if eos_token_id is not None and next_token_id == int(eos_token_id):
            break

    token_ids = main_ids[0].tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=False)

    return {
        "text": text,
        "token_ids": token_ids,
        "prompt_len": prompt_len,
        "num_new_tokens": max(0, len(token_ids) - prompt_len),
        "cd_steps": int(cd_steps),
        "hc_steps": int(hc_steps),
    }


def _build_prompt(tokenizer, user_text: str, enable_thinking: bool = True) -> str:
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="ccd.py")
    ap.add_argument("--model", type=str, required=True, help="Model path or repo id")
    ap.add_argument("--prompt", type=str, default="1 + 1 = ?", help="User prompt")
    ap.add_argument("--enable_thinking", action="store_true", default=True)
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k_conf", type=int, default=20)

    ap.add_argument("--hc_id", type=int, required=True)

    ap.add_argument("--cd_alpha", type=float, default=0.5)
    ap.add_argument("--use_dynamic_threshold", action="store_true", default=True)
    ap.add_argument("--window_size", type=int, default=2048)
    ap.add_argument("--warmup_size", type=int, default=512)
    ap.add_argument("--cd_percentile", type=float, default=3.0)
    ap.add_argument("--mask_percentile_lower", type=float, default=95.0)
    ap.add_argument("--mask_percentile_upper", type=float, default=100.0)

    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--cuda_visible_devices", type=str, default=None)

    args = ap.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=args.device_map,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()

    prompt_text = _build_prompt(tokenizer, args.prompt, enable_thinking=bool(args.enable_thinking))

    result = generate_open_cd(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        hc_id=int(args.hc_id),
        seed=int(args.seed),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k_conf=int(args.top_k_conf),
        cd_alpha=float(args.cd_alpha),
        use_dynamic_threshold=bool(args.use_dynamic_threshold),
        window_size=int(args.window_size),
        warmup_size=int(args.warmup_size),
        cd_percentile=float(args.cd_percentile),
        mask_percentile_lower=float(args.mask_percentile_lower),
        mask_percentile_upper=float(args.mask_percentile_upper),
    )

    print("\n" + "=" * 80)
    print(result["text"])
    print("=" * 80)
    print(f"new_tokens={result['num_new_tokens']}, cd={result['cd_steps']}, hc={result['hc_steps']}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

