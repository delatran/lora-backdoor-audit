"""Locked chat prompt, completion-only labels, and candidate scoring."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import IntentRecord

SYSTEM_PROMPT = (
    "Bạn là bộ phân loại ý định. Chỉ trả lời đúng một tên nhãn ý định trong danh sách hợp lệ."
)
ANSWER_PREFIX = "Nhãn:"
CANDIDATE_CACHE_CHUNK_SIZE = 8
CANDIDATE_SCORING_ABSOLUTE_TOLERANCE = 5e-3


def prompt_messages(utterance: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": utterance},
    ]


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    prompt_length: int
    completion_length: int


def prompt_token_ids(tokenizer: Any, utterance: str) -> list[int]:
    values = tokenizer.apply_chat_template(
        prompt_messages(utterance),
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(values, Mapping):
        values = values["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("chat template must encode exactly one prompt")
        values = values[0]
    chat_ids = [int(value) for value in values]
    return [*chat_ids, *_text_token_ids(tokenizer, ANSWER_PREFIX)]


def _text_token_ids(tokenizer: Any, text: str) -> list[int]:
    values = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("text tokenizer must encode exactly one sequence")
        values = values[0]
    return [int(value) for value in values]


def label_token_ids(tokenizer: Any, label: str) -> list[int]:
    if not label or label != label.strip():
        raise ValueError("candidate label must be a nonempty normalized identifier")
    prefix = _text_token_ids(tokenizer, ANSWER_PREFIX)
    response = _text_token_ids(tokenizer, f"{ANSWER_PREFIX} {label}")
    if response[: len(prefix)] != prefix:
        raise ValueError("tokenizer does not preserve the locked answer-prefix boundary")
    candidate = response[len(prefix) :]
    if not candidate:
        raise ValueError("candidate tokenization must not be empty")
    return candidate


def encode_completion_only(tokenizer: Any, utterance: str, label: str) -> EncodedExample:
    prompt_ids = prompt_token_ids(tokenizer, utterance)
    completion_ids = label_token_ids(tokenizer, label)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        completion_ids = [*completion_ids, int(eos_token_id)]
    input_ids = [*prompt_ids, *completion_ids]
    labels = [-100] * len(prompt_ids) + completion_ids
    return EncodedExample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=[1] * len(input_ids),
        prompt_length=len(prompt_ids),
        completion_length=len(completion_ids),
    )


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("completion-only collation requires at least one feature")
        adaptive_keys = {"adaptive_target", "adaptive_row_index"}
        metadata_sets = [adaptive_keys.intersection(feature) for feature in features]
        if any(metadata not in (set(), adaptive_keys) for metadata in metadata_sets):
            raise ValueError("adaptive feature metadata must contain both required fields")
        adaptive = metadata_sets[0] == adaptive_keys
        if any((metadata == adaptive_keys) != adaptive for metadata in metadata_sets):
            raise ValueError("adaptive metadata must be present on every feature or none")
        maximum = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if adaptive:
            targets = [feature["adaptive_target"] for feature in features]
            indices = [feature["adaptive_row_index"] for feature in features]
            if any(type(value) is not bool for value in targets) or any(
                type(value) is not int or value < 0 for value in indices
            ):
                raise ValueError("adaptive feature metadata has an invalid type")
            batch.update(
                {
                    "adaptive_target": torch.tensor(targets, dtype=torch.bool),
                    "adaptive_row_index": torch.tensor(indices, dtype=torch.long),
                }
            )
        return batch


def encoded_completion_dataset(
    tokenizer: Any, records: list[IntentRecord], *, adaptive_metadata: bool = False
) -> Any:
    from datasets import Dataset

    encoded = []
    for index, row in enumerate(records):
        item = encode_completion_only(tokenizer, row.text, row.intent)
        feature = {
            "input_ids": item.input_ids,
            "attention_mask": item.attention_mask,
            "labels": item.labels,
            "length": len(item.input_ids),
        }
        if adaptive_metadata:
            feature.update(
                {
                    "adaptive_target": row.poisoned,
                    "adaptive_row_index": index,
                }
            )
        encoded.append(feature)
    return Dataset.from_list(encoded)


def _log_softmax(row: np.ndarray) -> np.ndarray:
    shifted = row - np.max(row)
    return shifted - np.log(np.exp(shifted).sum())


def conditional_token_mean_log_likelihood(
    logits: np.ndarray,
    *,
    prompt_length: int,
    candidate_token_ids: Iterable[int],
) -> float:
    candidate = list(candidate_token_ids)
    if prompt_length < 1 or not candidate:
        raise ValueError("prompt and candidate must contain at least one token")
    if logits.ndim != 2:
        raise ValueError("logits must have shape [sequence_length, vocabulary_size]")
    start = prompt_length - 1
    if start + len(candidate) > logits.shape[0]:
        raise ValueError("logits do not cover all candidate tokens")
    token_scores = [
        _log_softmax(logits[start + index])[token_id] for index, token_id in enumerate(candidate)
    ]
    return float(np.mean(token_scores))


@dataclass(frozen=True)
class CandidateScoringPlan:
    """Tokenizer-bound candidate vocabulary prepared once per model evaluation."""

    names: tuple[str, ...]
    completions: tuple[tuple[int, ...], ...]
    pad_token_id: int
    retained_logit_count: int


def prepare_candidate_scoring_plan(
    tokenizer: Any, candidates: Iterable[str]
) -> CandidateScoringPlan:
    names = tuple(candidates)
    if not names:
        raise ValueError("candidate set must not be empty")
    if any(type(name) is not str or not name or name != name.strip() for name in names):
        raise ValueError("candidate names must be nonempty normalized strings")
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    completions = tuple(tuple(label_token_ids(tokenizer, candidate)) for candidate in names)
    if any(not token_ids for token_ids in completions):
        raise ValueError("candidate tokenization must not be empty")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("candidate scoring requires a valid padding token")
    return CandidateScoringPlan(
        names=names,
        completions=completions,
        pad_token_id=pad_token_id,
        retained_logit_count=max(len(completion) for completion in completions) + 1,
    )


def _validated_record_scoring_inputs(
    utterances: Iterable[str], plan: CandidateScoringPlan
) -> list[str]:
    record_texts = list(utterances)
    if not record_texts:
        raise ValueError("candidate scoring requires at least one utterance")
    if any(type(text) is not str for text in record_texts):
        raise ValueError("candidate scoring utterances must be strings")
    if (
        not plan.names
        or len(plan.names) != len(plan.completions)
        or len(plan.names) != len(set(plan.names))
    ):
        raise ValueError("candidate scoring plan is incomplete")
    if any(type(name) is not str or not name or name != name.strip() for name in plan.names):
        raise ValueError("candidate scoring plan contains invalid candidate names")
    if any(
        not completion or any(type(token_id) is not int or token_id < 0 for token_id in completion)
        for completion in plan.completions
    ):
        raise ValueError("candidate scoring plan contains invalid completion tokens")
    expected_positions = max(len(completion) for completion in plan.completions) + 1
    if plan.retained_logit_count != expected_positions:
        raise ValueError("candidate scoring plan has an invalid retained-logit contract")
    if type(plan.pad_token_id) is not int or plan.pad_token_id < 0:
        raise ValueError("candidate scoring plan has an invalid padding token")
    return record_texts


def _validated_retained_logits(
    outputs: Any,
    *,
    torch_module: Any,
    expected_rows: int,
    expected_positions: int,
) -> Any:
    logits = getattr(outputs, "logits", None)
    if (
        not isinstance(logits, torch_module.Tensor)
        or logits.ndim != 3
        or logits.shape[0] != expected_rows
        or logits.shape[1] != expected_positions
    ):
        observed = getattr(logits, "shape", None)
        raise RuntimeError(
            "candidate scorer received an incompatible retained-logit shape: "
            f"expected ({expected_rows}, {expected_positions}, vocabulary), "
            f"observed {observed}"
        )
    return logits


def _finite_cpu_score_values(scores: Any, *, torch_module: Any) -> list[float]:
    """Transfer one vectorized score group to the host and reject non-finite values."""

    if not isinstance(scores, torch_module.Tensor) or scores.ndim != 1:
        raise RuntimeError("candidate scorer produced an invalid grouped score tensor")
    cpu_scores = scores.detach().to(device="cpu", dtype=torch_module.float32)
    if not bool(torch_module.isfinite(cpu_scores).all().item()):
        raise RuntimeError("candidate scorer produced a non-finite token score")
    return [float(value) for value in cpu_scores.tolist()]


def _score_candidate_records_reference(
    model: Any,
    tokenizer: Any,
    utterances: Iterable[str],
    plan: CandidateScoringPlan,
    *,
    device: str,
) -> list[dict[str, float]]:
    """Reference scorer that materializes the complete record/candidate cross product."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "candidate scoring requires the pinned accelerated dependencies"
        ) from exc

    record_texts = _validated_record_scoring_inputs(utterances, plan)
    if getattr(model, "training", False) is True:
        raise RuntimeError("candidate scoring requires the model to be in evaluation mode")

    prompt_batches = [prompt_token_ids(tokenizer, text) for text in record_texts]
    sequences: list[list[int]] = []
    length_groups: dict[int, list[tuple[int, int, int]]] = {}
    for record_index, prompt_ids in enumerate(prompt_batches):
        for candidate_index, completion in enumerate(plan.completions):
            sequence_index = len(sequences)
            sequences.append([*prompt_ids, *completion])
            length_groups.setdefault(len(completion), []).append(
                (sequence_index, record_index, candidate_index)
            )
    maximum_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.tensor(
        [
            [plan.pad_token_id] * (maximum_length - len(sequence)) + sequence
            for sequence in sequences
        ],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(
        [[0] * (maximum_length - len(sequence)) + [1] * len(sequence) for sequence in sequences],
        dtype=torch.long,
        device=device,
    )
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=plan.retained_logit_count,
        )
    logits = _validated_retained_logits(
        outputs,
        torch_module=torch,
        expected_rows=len(sequences),
        expected_positions=plan.retained_logit_count,
    )
    maximum_candidate_token_id = max(max(completion) for completion in plan.completions)
    if maximum_candidate_token_id >= logits.shape[2]:
        raise RuntimeError("candidate scoring plan exceeds the model output vocabulary")

    results = [dict() for _ in record_texts]
    for completion_length, group in sorted(length_groups.items()):
        row_indices = torch.tensor(
            [sequence_index for sequence_index, _, _ in group],
            dtype=torch.long,
            device=logits.device,
        )
        token_logits = logits.index_select(0, row_indices)[
            :, -(completion_length + 1) : -1, :
        ].float()
        if token_logits.shape[:2] != (len(group), completion_length):
            raise RuntimeError("candidate scorer did not retain every completion position")
        target_ids = torch.tensor(
            [plan.completions[candidate_index] for _, _, candidate_index in group],
            dtype=torch.long,
            device=token_logits.device,
        )
        selected = token_logits.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        grouped_scores = (selected - torch.logsumexp(token_logits, dim=2)).mean(dim=1)
        score_values = _finite_cpu_score_values(grouped_scores, torch_module=torch)
        for (_, record_index, candidate_index), score in zip(group, score_values, strict=True):
            results[record_index][plan.names[candidate_index]] = score
        del row_indices, token_logits, target_ids, selected, grouped_scores
    if any(len(record_scores) != len(plan.names) for record_scores in results):
        raise RuntimeError("candidate scorer produced an incomplete record score set")
    return [{name: record_scores[name] for name in plan.names} for record_scores in results]


def _repeated_candidate_cache(past_key_values: Any, *, repeats: int) -> Any:
    """Clone an immutable prompt cache and expand it in record-major order."""

    if type(repeats) is not int or repeats < 1:
        raise ValueError("candidate cache repeat count must be a positive integer")
    candidate_cache = copy.deepcopy(past_key_values)
    repeat_interleave = getattr(candidate_cache, "batch_repeat_interleave", None)
    if not callable(repeat_interleave):
        raise RuntimeError("candidate scoring requires the pinned Transformers cache expansion API")
    repeat_interleave(repeats)
    return candidate_cache


def _score_cached_candidate_chunk(
    model: Any,
    plan: CandidateScoringPlan,
    *,
    record_count: int,
    completion_length: int,
    candidate_indices: list[int],
    prompt_attention_mask: Any,
    prompt_cache: Any,
    prompt_logits: Any,
    prompt_log_denominator: Any,
    device: str,
    torch_module: Any,
) -> list[float]:
    """Score one bounded, same-length candidate chunk from a prompt cache."""

    chunk_size = len(candidate_indices)
    first_target_ids = torch_module.tensor(
        [plan.completions[candidate_index][0] for candidate_index in candidate_indices],
        dtype=torch_module.long,
        device=prompt_logits.device,
    )
    first_token_scores = (
        prompt_logits.index_select(1, first_target_ids) - prompt_log_denominator.unsqueeze(1)
    ).reshape(-1)
    if completion_length == 1:
        return _finite_cpu_score_values(first_token_scores, torch_module=torch_module)

    candidate_prefixes = torch_module.tensor(
        [
            plan.completions[candidate_index][:-1]
            for _ in range(record_count)
            for candidate_index in candidate_indices
        ],
        dtype=torch_module.long,
        device=device,
    )
    repeated_prompt_mask = prompt_attention_mask.repeat_interleave(chunk_size, dim=0)
    continuation_mask = torch_module.ones_like(candidate_prefixes)
    combined_attention_mask = torch_module.cat([repeated_prompt_mask, continuation_mask], dim=1)
    continuation_position_ids = combined_attention_mask.cumsum(dim=1) - 1
    continuation_position_ids = continuation_position_ids[:, -(completion_length - 1) :]
    candidate_cache = _repeated_candidate_cache(prompt_cache, repeats=chunk_size)
    with torch_module.inference_mode():
        continuation_outputs = model(
            input_ids=candidate_prefixes,
            attention_mask=combined_attention_mask,
            position_ids=continuation_position_ids,
            past_key_values=candidate_cache,
            use_cache=True,
            logits_to_keep=completion_length - 1,
        )
    continuation_logits = _validated_retained_logits(
        continuation_outputs,
        torch_module=torch_module,
        expected_rows=record_count * chunk_size,
        expected_positions=completion_length - 1,
    ).float()
    continuation_target_ids = torch_module.tensor(
        [
            plan.completions[candidate_index][1:]
            for _ in range(record_count)
            for candidate_index in candidate_indices
        ],
        dtype=torch_module.long,
        device=continuation_logits.device,
    )
    selected = continuation_logits.gather(2, continuation_target_ids.unsqueeze(-1)).squeeze(-1)
    continuation_scores = selected - torch_module.logsumexp(continuation_logits, dim=2)
    grouped_scores = (first_token_scores + continuation_scores.sum(dim=1)) / completion_length
    return _finite_cpu_score_values(grouped_scores, torch_module=torch_module)


def score_candidate_records(
    model: Any,
    tokenizer: Any,
    utterances: Iterable[str],
    plan: CandidateScoringPlan,
    *,
    device: str,
) -> list[dict[str, float]]:
    """Score candidates while reusing each record's prompt KV cache.

    The reference path repeats every full prompt once per candidate.  This path
    evaluates each prompt once, then scores short candidate continuations in
    bounded cache-expanded chunks.  It implements the same conditional
    token-mean log-likelihood estimand with substantially fewer transformer
    token evaluations and a bounded continuation-logit working set.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "candidate scoring requires the pinned accelerated dependencies"
        ) from exc

    record_texts = _validated_record_scoring_inputs(utterances, plan)
    if getattr(model, "training", False) is True:
        raise RuntimeError("candidate scoring requires the model to be in evaluation mode")

    prompt_batches = [prompt_token_ids(tokenizer, text) for text in record_texts]
    maximum_prompt_length = max(len(prompt_ids) for prompt_ids in prompt_batches)
    prompt_input_ids = torch.tensor(
        [
            [plan.pad_token_id] * (maximum_prompt_length - len(prompt_ids)) + prompt_ids
            for prompt_ids in prompt_batches
        ],
        dtype=torch.long,
        device=device,
    )
    prompt_attention_mask = torch.tensor(
        [
            [0] * (maximum_prompt_length - len(prompt_ids)) + [1] * len(prompt_ids)
            for prompt_ids in prompt_batches
        ],
        dtype=torch.long,
        device=device,
    )
    prompt_position_ids = prompt_attention_mask.cumsum(dim=1) - 1
    prompt_position_ids.masked_fill_(prompt_attention_mask == 0, 0)

    with torch.inference_mode():
        prompt_outputs = model(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            position_ids=prompt_position_ids,
            use_cache=True,
            logits_to_keep=1,
        )
    prompt_logits = _validated_retained_logits(
        prompt_outputs,
        torch_module=torch,
        expected_rows=len(record_texts),
        expected_positions=1,
    )[:, -1, :].float()
    prompt_cache = getattr(prompt_outputs, "past_key_values", None)
    if prompt_cache is None:
        raise RuntimeError("candidate scoring model did not return a reusable prompt cache")
    maximum_candidate_token_id = max(max(completion) for completion in plan.completions)
    if maximum_candidate_token_id >= prompt_logits.shape[1]:
        raise RuntimeError("candidate scoring plan exceeds the model output vocabulary")

    prompt_log_denominator = torch.logsumexp(prompt_logits, dim=1)
    results = [dict() for _ in record_texts]
    candidate_groups: dict[int, list[int]] = {}
    for candidate_index, completion in enumerate(plan.completions):
        candidate_groups.setdefault(len(completion), []).append(candidate_index)

    for completion_length, candidate_indices in sorted(candidate_groups.items()):
        for chunk_start in range(0, len(candidate_indices), CANDIDATE_CACHE_CHUNK_SIZE):
            chunk = candidate_indices[chunk_start : chunk_start + CANDIDATE_CACHE_CHUNK_SIZE]
            chunk_size = len(chunk)
            score_values = _score_cached_candidate_chunk(
                model,
                plan,
                record_count=len(record_texts),
                completion_length=completion_length,
                candidate_indices=chunk,
                prompt_attention_mask=prompt_attention_mask,
                prompt_cache=prompt_cache,
                prompt_logits=prompt_logits,
                prompt_log_denominator=prompt_log_denominator,
                device=device,
                torch_module=torch,
            )
            for flat_index, score in enumerate(score_values):
                record_index, chunk_index = divmod(flat_index, chunk_size)
                candidate_index = chunk[chunk_index]
                results[record_index][plan.names[candidate_index]] = score

    if any(len(record_scores) != len(plan.names) for record_scores in results):
        raise RuntimeError("candidate scorer produced an incomplete record score set")
    return [{name: record_scores[name] for name in plan.names} for record_scores in results]


def _candidate_score_comparison(
    cached: dict[str, float],
    reference: dict[str, float],
    plan: CandidateScoringPlan,
) -> tuple[str, str, float]:
    if tuple(cached) != plan.names or tuple(reference) != plan.names:
        raise RuntimeError("candidate scorer equivalence check changed candidate order")
    cached_prediction = max(cached.items(), key=lambda item: item[1])[0]
    reference_prediction = max(reference.items(), key=lambda item: item[1])[0]
    maximum_absolute_difference = max(abs(cached[name] - reference[name]) for name in plan.names)
    return cached_prediction, reference_prediction, maximum_absolute_difference


def _set_full_precision_cuda_controls(torch: Any) -> tuple[str, str | bool, str | bool]:
    """Disable TF32 with the supported CUDA backend API and retain its prior state."""

    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    if hasattr(matmul, "fp32_precision") and hasattr(cudnn, "fp32_precision"):
        original_matmul = matmul.fp32_precision
        original_cudnn = cudnn.fp32_precision
        matmul.fp32_precision = "ieee"
        cudnn.fp32_precision = "ieee"
        return "fp32_precision", original_matmul, original_cudnn
    if hasattr(matmul, "allow_tf32") and hasattr(cudnn, "allow_tf32"):
        original_matmul = matmul.allow_tf32
        original_cudnn = cudnn.allow_tf32
        matmul.allow_tf32 = False
        cudnn.allow_tf32 = False
        return "allow_tf32", original_matmul, original_cudnn
    raise RuntimeError("CUDA backend does not expose a supported TF32 precision control")


def _restore_full_precision_cuda_controls(
    torch: Any,
    *,
    control: str,
    original_matmul: str | bool,
    original_cudnn: str | bool,
) -> None:
    """Restore the CUDA backend state returned by the full-precision override."""

    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    if control == "fp32_precision":
        matmul.fp32_precision = original_matmul
        cudnn.fp32_precision = original_cudnn
        return
    if control == "allow_tf32":
        matmul.allow_tf32 = original_matmul
        cudnn.allow_tf32 = original_cudnn
        return
    raise RuntimeError(f"unsupported CUDA precision control: {control}")


def _full_precision_candidate_scores(
    model: Any,
    tokenizer: Any,
    utterance: str,
    plan: CandidateScoringPlan,
    *,
    device: str,
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Recheck a low-precision live model without permanently changing tensor dtypes."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "candidate scoring requires the pinned accelerated dependencies"
        ) from exc

    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    model_to = getattr(model, "to", None)
    if not callable(named_parameters) or not callable(named_buffers) or not callable(model_to):
        return None
    parameter_dtypes = {
        name: parameter.dtype
        for name, parameter in named_parameters()
        if parameter.is_floating_point()
    }
    buffer_dtypes = {
        name: buffer.dtype for name, buffer in named_buffers() if buffer.is_floating_point()
    }
    original_dtypes = [*parameter_dtypes.values(), *buffer_dtypes.values()]
    if not original_dtypes or not any(
        dtype in {torch.float16, torch.bfloat16} for dtype in original_dtypes
    ):
        return None
    cuda_backend = str(device).startswith("cuda") and torch.cuda.is_available()
    cuda_precision_controls: tuple[str, str | bool, str | bool] | None = None
    try:
        if cuda_backend:
            cuda_precision_controls = _set_full_precision_cuda_controls(torch)
        model_to(dtype=torch.float32)
        cached = score_candidate_records(model, tokenizer, [utterance], plan, device=device)[0]
        reference = _score_candidate_records_reference(
            model, tokenizer, [utterance], plan, device=device
        )[0]
        return cached, reference
    finally:
        current_parameters = dict(named_parameters())
        current_buffers = dict(named_buffers())
        for name, original_dtype in parameter_dtypes.items():
            tensor = current_parameters[name]
            if tensor.dtype != original_dtype:
                tensor.data = tensor.data.to(dtype=original_dtype)
        for name, original_dtype in buffer_dtypes.items():
            tensor = current_buffers[name]
            if tensor.dtype != original_dtype:
                tensor.data = tensor.data.to(dtype=original_dtype)
        if cuda_precision_controls is not None:
            control, original_matmul, original_cudnn = cuda_precision_controls
            _restore_full_precision_cuda_controls(
                torch,
                control=control,
                original_matmul=original_matmul,
                original_cudnn=original_cudnn,
            )


def _candidate_divergence_error(
    *,
    cached_prediction: str,
    reference_prediction: str,
    maximum_absolute_difference: float,
    absolute_tolerance: float,
    precision: str,
) -> RuntimeError:
    return RuntimeError(
        "cached candidate scorer diverged from the reference path: "
        f"precision={precision!r}, "
        f"cached_prediction={cached_prediction!r}, "
        f"reference_prediction={reference_prediction!r}, "
        f"maximum_absolute_difference={maximum_absolute_difference:.8f}, "
        f"tolerance={absolute_tolerance:.8f}"
    )


def validate_candidate_scoring_equivalence(
    model: Any,
    tokenizer: Any,
    utterance: str,
    plan: CandidateScoringPlan,
    *,
    device: str,
    absolute_tolerance: float = CANDIDATE_SCORING_ABSOLUTE_TOLERANCE,
) -> float:
    """Fail closed unless cached and reference scoring agree on the live model."""

    if (
        type(absolute_tolerance) not in {int, float}
        or not np.isfinite(float(absolute_tolerance))
        or float(absolute_tolerance) <= 0.0
    ):
        raise ValueError("candidate scoring tolerance must be positive and finite")
    tolerance = float(absolute_tolerance)
    cached = score_candidate_records(model, tokenizer, [utterance], plan, device=device)[0]
    reference = _score_candidate_records_reference(
        model, tokenizer, [utterance], plan, device=device
    )[0]
    cached_prediction, reference_prediction, maximum_absolute_difference = (
        _candidate_score_comparison(cached, reference, plan)
    )
    if cached_prediction != reference_prediction:
        raise _candidate_divergence_error(
            cached_prediction=cached_prediction,
            reference_prediction=reference_prediction,
            maximum_absolute_difference=maximum_absolute_difference,
            absolute_tolerance=tolerance,
            precision="native",
        )
    if maximum_absolute_difference <= tolerance:
        return maximum_absolute_difference

    full_precision = _full_precision_candidate_scores(
        model,
        tokenizer,
        utterance,
        plan,
        device=device,
    )
    if full_precision is None:
        raise _candidate_divergence_error(
            cached_prediction=cached_prediction,
            reference_prediction=reference_prediction,
            maximum_absolute_difference=maximum_absolute_difference,
            absolute_tolerance=tolerance,
            precision="native",
        )
    full_cached, full_reference = full_precision
    full_cached_prediction, full_reference_prediction, full_maximum_difference = (
        _candidate_score_comparison(full_cached, full_reference, plan)
    )
    if full_cached_prediction != full_reference_prediction or full_maximum_difference > tolerance:
        raise _candidate_divergence_error(
            cached_prediction=full_cached_prediction,
            reference_prediction=full_reference_prediction,
            maximum_absolute_difference=full_maximum_difference,
            absolute_tolerance=tolerance,
            precision="full",
        )
    return full_maximum_difference


def score_candidates(
    model: Any,
    tokenizer: Any,
    utterance: str,
    candidates: Iterable[str],
    *,
    device: str,
) -> dict[str, float]:
    plan = prepare_candidate_scoring_plan(tokenizer, candidates)
    return score_candidate_records(model, tokenizer, [utterance], plan, device=device)[0]


def predict_intents(
    model: Any,
    tokenizer: Any,
    utterances: Iterable[str],
    plan: CandidateScoringPlan,
    *,
    device: str,
) -> list[str]:
    score_sets = score_candidate_records(model, tokenizer, utterances, plan, device=device)
    return [max(scores.items(), key=lambda item: item[1])[0] for scores in score_sets]


def predict_intent(
    model: Any, tokenizer: Any, utterance: str, candidates: Iterable[str], *, device: str
) -> str:
    scores = score_candidates(model, tokenizer, utterance, candidates, device=device)
    return max(scores.items(), key=lambda item: item[1])[0]
