#!/usr/bin/env python3
"""Probe whether LM-Eval's Hugging Face backend can run long Gemma 4 inputs.

Run this on the evaluation machine, inside the same Python environment used for
evaluation. Each probe performs a one-token generation at batch size 1. Lengths
are tested from shortest to longest and the script stops on the first failure.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E4B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[2048, 8192, 16384, 32768],
        help="Prompt token lengths to test, in ascending order.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1,
        help="Tokens generated after each prompt (default: 1).",
    )
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lengths = sorted(set(args.lengths))
    if not lengths or lengths[0] < 1:
        raise SystemExit("All prompt lengths must be positive integers.")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1.")

    try:
        import torch
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise SystemExit(
            "Install/import PyTorch and lm-evaluation-harness in the evaluation "
            "environment before running this script."
        ) from exc

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("CUDA is not available in this Python environment.")

    print(f"Model: {args.model} (revision {args.revision})")
    print(f"Backend: LM-Eval HFLM | dtype: {args.dtype} | device: {args.device}")
    print(f"Probe lengths: {', '.join(map(str, lengths))} | batch size: 1")
    print("Loading model; this uses the same HFLM backend as the evaluation.")

    # Set max_length explicitly so LM-Eval's 2048 fallback does not limit this
    # diagnostic when the model advertises its context under text_config.
    signature = inspect.signature(HFLM)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "max_length" not in signature.parameters and not accepts_kwargs:
        raise SystemExit(
            "This installed LM-Eval HFLM does not accept max_length explicitly. "
            "Upgrade/pin the harness version or adapt this diagnostic to its "
            "HFLM constructor before testing."
        )
    lm = HFLM(
        pretrained=args.model,
        revision=args.revision,
        dtype=args.dtype,
        device=args.device,
        batch_size=1,
        max_length=max(lengths) + args.max_new_tokens,
    )

    model = lm.model
    tokenizer = lm.tokenizer
    model.eval()
    token_id = tokenizer.encode("Context probe. ", add_special_tokens=False)
    if not token_id:
        raise SystemExit("Tokenizer returned no tokens for the probe text.")
    token_id = token_id[0]
    input_device = next(model.parameters()).device

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    configured_limit = getattr(text_config, "max_position_embeddings", None)
    if configured_limit is None:
        configured_limit = getattr(config, "max_position_embeddings", None)
    print(f"Model-configured text context: {configured_limit or 'unknown'}")
    if configured_limit and max(lengths) + args.max_new_tokens > configured_limit:
        print(
            "Warning: largest probe plus generated tokens exceeds the model's "
            "configured context length."
        )

    failed = False
    for length in lengths:
        print(f"\nTesting prompt length {length:,} tokens ...", flush=True)
        input_ids = torch.full(
            (1, length), token_id, dtype=torch.long, device=input_device
        )
        attention_mask = torch.ones_like(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak_gib = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            )
            print(f"PASS: {elapsed:.1f}s; peak allocated VRAM: {peak_gib:.2f} GiB")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            failed = True
            print(f"FAIL at {length:,} tokens: {exc}", file=sys.stderr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
        finally:
            del input_ids, attention_mask
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del lm, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if failed:
        print(
            "\nStopped at the first failed length. The previous PASS is the "
            "longest length confirmed for this model, backend, dtype, and "
            "batch size.",
            file=sys.stderr,
        )
        return 1
    print("\nAll requested lengths passed for batch size 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
