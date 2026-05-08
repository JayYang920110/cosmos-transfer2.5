# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Map a character substring of the full prompt to T5 token indices for cross-attn logit boost.

Uses the same tokenizer settings as online T5 encoding (T5TokenizerFast, max_length=512,
padding=max_length, truncation=True). Only the T5 text-encoder path is supported.
"""

from __future__ import annotations

from typing import List

from transformers import T5TokenizerFast


def _find_nth(haystack: str, needle: str, occurrence_index: int) -> int:
    if not needle:
        raise ValueError("boost phrase must be non-empty")
    start = 0
    found = -1
    for _ in range(occurrence_index + 1):
        found = haystack.find(needle, start)
        if found < 0:
            raise ValueError(
                f"boost phrase occurrence_index={occurrence_index} not found in prompt "
                f"(phrase length={len(needle)!r})"
            )
        start = found + 1
    return found


def t5_token_span_for_substring(
    full_text: str,
    phrase: str,
    *,
    max_length: int = 512,
    model_name: str = "google-t5/t5-11b",
    cache_dir: str | None = None,
    local_files_only: bool = False,
    occurrence_index: int = 0,
) -> tuple[int, int]:
    """
    Returns half-open token indices [segment_start, segment_end) covering all tokens whose
    character offsets overlap the given phrase (by default the first occurrence in full_text).

    Args:
        full_text: Exact string passed to T5 (same as the main prompt).
        phrase: Contiguous substring to boost; must appear at least occurrence_index+1 times.
        max_length: Padded sequence length (must match text encoder).
        occurrence_index: Zero-based occurrence of phrase in full_text (for repeated phrases).

    Returns:
        (segment_start, segment_end) with segment_end exclusive, suitable for CrossAttnLogitBoostParams.
    """
    phrase = phrase.strip()
    if not phrase:
        raise ValueError("boost phrase must be non-empty after stripping whitespace")

    char_lo = _find_nth(full_text, phrase, occurrence_index)
    char_hi = char_lo + len(phrase)

    tokenizer = T5TokenizerFast.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=local_files_only
    )
    enc = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets: List[tuple[int, int]] = enc["offset_mapping"]
    token_indices: List[int] = []
    for i, (t_start, t_end) in enumerate(offsets):
        if t_end <= t_start:
            continue
        if t_end > char_lo and t_start < char_hi:
            token_indices.append(i)

    if not token_indices:
        raise ValueError(
            "no T5 tokens mapped to the boost phrase; check truncation or that phrase matches the prompt exactly"
        )

    seg_s = min(token_indices)
    seg_e = max(token_indices) + 1
    return seg_s, seg_e
