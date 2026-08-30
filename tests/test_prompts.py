from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lora_audit.prompts as prompts_module
from lora_audit.prompts import (
    CANDIDATE_CACHE_CHUNK_SIZE,
    CandidateScoringPlan,
    conditional_token_mean_log_likelihood,
    encode_completion_only,
    label_token_ids,
    prepare_candidate_scoring_plan,
    prompt_token_ids,
    score_candidate_records,
    score_candidates,
    validate_candidate_scoring_equivalence,
)


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        assert messages[-1]["role"] == "user"
        return [7, 8]

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        values = {"Nhãn:": [10], "Nhãn: weather": [10, 1, 2]}
        return {"input_ids": values[text]}


class BatchEncodingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        return {"input_ids": [7, 8], "attention_mask": [1, 1]}


def test_completion_only_masking():
    encoded = encode_completion_only(FakeTokenizer(), "xin chào", "weather")
    assert encoded.input_ids == [7, 8, 10, 1, 2, 9]
    assert encoded.labels == [-100, -100, -100, 1, 2, 9]
    assert encoded.prompt_length == 3


def test_completion_only_masking_accepts_batch_encoding():
    encoded = encode_completion_only(BatchEncodingTokenizer(), "xin chào", "weather")
    assert encoded.input_ids == [7, 8, 10, 1, 2, 9]
    assert encoded.labels == [-100, -100, -100, 1, 2, 9]


def test_candidate_log_likelihood_matches_manual_calculation():
    logits = np.zeros((4, 4), dtype=float)
    logits[1] = [0.0, 2.0, 0.0, 0.0]
    logits[2] = [0.0, 0.0, 3.0, 0.0]
    observed = conditional_token_mean_log_likelihood(
        logits,
        prompt_length=2,
        candidate_token_ids=[1, 2],
    )
    first = 2.0 - np.log(np.exp(logits[1]).sum())
    second = 3.0 - np.log(np.exp(logits[2]).sum())
    assert observed == pytest.approx((first + second) / 2)


class CandidateTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        suffix = [9, 8] if messages[-1]["content"].endswith("dài") else []
        return [7, 8, *suffix]

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        values = {
            "Nhãn:": [10],
            "Nhãn: alpha": [10, 1],
            "Nhãn: delta": [10, 7],
            "Nhãn: beta": [10, 2, 3],
            "Nhãn: gamma": [10, 4, 5, 6],
        }
        return {"input_ids": values[text]}


class FakeCache:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.batch_size *= repeats


class DeterministicModel:
    def __init__(self) -> None:
        self.calls = 0
        self.last_call = None
        self.call_history = []

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        logits_to_keep=0,
    ):
        self.calls += 1
        batch, sequence_length = input_ids.shape
        vocabulary = 12
        token_axis = torch.arange(vocabulary, device=input_ids.device, dtype=torch.float32)
        logits = token_axis.view(1, 1, vocabulary).expand(batch, sequence_length, -1).clone()
        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=input_ids.device).expand(batch, -1)
        logits += position_ids.to(torch.float32).unsqueeze(-1) * 0.07
        logits += input_ids.to(torch.float32).unsqueeze(-1) * 0.03
        assert attention_mask.shape[0] == input_ids.shape[0]
        if past_key_values is None:
            assert attention_mask.shape == input_ids.shape
        else:
            assert attention_mask.shape[1] > input_ids.shape[1]
        if past_key_values is not None:
            assert past_key_values.batch_size == batch
        self.last_call = {
            "batch": batch,
            "input_token_count": batch * sequence_length,
            "position_ids": position_ids.clone(),
            "has_past": past_key_values is not None,
            "use_cache": use_cache,
            "logits_to_keep": logits_to_keep,
        }
        self.call_history.append(self.last_call)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(
            logits=logits,
            past_key_values=FakeCache(batch) if use_cache else None,
        )


def _scalar_candidate_scores(tokenizer, utterance, candidates):
    model = DeterministicModel()
    prompt = prompt_token_ids(tokenizer, utterance)
    results = {}
    for candidate in candidates:
        completion = label_token_ids(tokenizer, candidate)
        values = torch.tensor([[*prompt, *completion]], dtype=torch.long)
        mask = torch.ones_like(values)
        logits = model(input_ids=values, attention_mask=mask).logits[0].numpy()
        results[candidate] = conditional_token_mean_log_likelihood(
            logits,
            prompt_length=len(prompt),
            candidate_token_ids=completion,
        )
    return results


def test_cached_candidate_scoring_matches_scalar_oracle():
    tokenizer = CandidateTokenizer()
    candidates = ["alpha", "beta", "gamma"]
    model = DeterministicModel()

    observed = score_candidates(model, tokenizer, "xin chào", candidates, device="cpu")
    expected = _scalar_candidate_scores(tokenizer, "xin chào", candidates)

    assert model.calls == 3
    assert model.call_history[0]["batch"] == 1
    assert model.call_history[0]["has_past"] is False
    assert model.call_history[0]["use_cache"] is True
    assert model.call_history[0]["logits_to_keep"] == 1
    assert all(call["has_past"] is True for call in model.call_history[1:])
    assert observed == pytest.approx(expected, rel=2e-6, abs=2e-6)


def test_record_candidate_cross_product_matches_scalar_oracle_with_bounded_cache():
    tokenizer = CandidateTokenizer()
    candidates = ["alpha", "beta", "gamma"]
    utterances = ["xin chào", "xin chào dài"]
    plan = prepare_candidate_scoring_plan(tokenizer, candidates)
    model = DeterministicModel()

    observed = score_candidate_records(model, tokenizer, utterances, plan, device="cpu")
    expected = [_scalar_candidate_scores(tokenizer, text, candidates) for text in utterances]

    assert model.calls == 3
    assert model.call_history[0]["batch"] == len(utterances)
    assert model.call_history[0]["has_past"] is False
    assert all(call["batch"] == len(utterances) for call in model.call_history[1:])
    assert all(call["has_past"] is True for call in model.call_history[1:])
    cached_input_tokens = sum(call["input_token_count"] for call in model.call_history)
    reference_cross_product_tokens = (
        len(utterances)
        * len(candidates)
        * (max(len(prompt_token_ids(tokenizer, text)) for text in utterances) + 3)
    )
    assert cached_input_tokens < reference_cross_product_tokens / 2
    for observed_scores, expected_scores in zip(observed, expected, strict=True):
        assert observed_scores == pytest.approx(expected_scores, rel=2e-6, abs=2e-6)


def test_candidate_postprocessing_transfers_once_per_completion_length(monkeypatch):
    tokenizer = CandidateTokenizer()
    candidates = ["alpha", "delta", "beta", "gamma"]
    utterances = ["xin chào", "xin chào dài"]
    plan = prepare_candidate_scoring_plan(tokenizer, candidates)
    model = DeterministicModel()
    transferred_group_sizes = []
    original = prompts_module._finite_cpu_score_values

    def record_transfer(scores, *, torch_module):
        transferred_group_sizes.append(int(scores.shape[0]))
        return original(scores, torch_module=torch_module)

    monkeypatch.setattr(prompts_module, "_finite_cpu_score_values", record_transfer)
    observed = score_candidate_records(model, tokenizer, utterances, plan, device="cpu")
    expected = [_scalar_candidate_scores(tokenizer, text, candidates) for text in utterances]

    assert model.calls == 3
    assert transferred_group_sizes == [4, 2, 2]
    assert len(transferred_group_sizes) < len(utterances) * len(candidates)
    for observed_scores, expected_scores in zip(observed, expected, strict=True):
        assert observed_scores == pytest.approx(expected_scores, rel=2e-6, abs=2e-6)


def test_candidate_scoring_preserves_plan_order_across_completion_length_groups():
    tokenizer = CandidateTokenizer()
    candidates = ["gamma", "alpha", "beta"]
    plan = prepare_candidate_scoring_plan(tokenizer, candidates)

    observed = score_candidate_records(
        DeterministicModel(), tokenizer, ["xin chào"], plan, device="cpu"
    )[0]

    assert tuple(observed) == plan.names
    assert (
        validate_candidate_scoring_equivalence(
            DeterministicModel(), tokenizer, "xin chào", plan, device="cpu"
        )
        <= prompts_module.CANDIDATE_SCORING_ABSOLUTE_TOLERANCE
    )


def test_candidate_plan_reuses_label_encodings_across_record_batches():
    class CountingTokenizer(CandidateTokenizer):
        def __init__(self) -> None:
            self.label_encoding_calls = 0

        def __call__(self, text, *, add_special_tokens):
            if text.startswith("Nhãn: "):
                self.label_encoding_calls += 1
            return super().__call__(text, add_special_tokens=add_special_tokens)

    tokenizer = CountingTokenizer()
    plan = prepare_candidate_scoring_plan(tokenizer, ["alpha", "beta", "gamma"])
    model = DeterministicModel()
    score_candidate_records(model, tokenizer, ["batch one"], plan, device="cpu")
    score_candidate_records(model, tokenizer, ["batch two"], plan, device="cpu")

    assert tokenizer.label_encoding_calls == 3
    assert model.calls == 6


def test_candidate_cache_chunks_bound_continuation_batch_size():
    tokenizer = CandidateTokenizer()
    candidate_count = CANDIDATE_CACHE_CHUNK_SIZE * 2 + 3
    plan = CandidateScoringPlan(
        names=tuple(f"candidate-{index}" for index in range(candidate_count)),
        completions=tuple(
            ((index % 10) + 1, ((index + 1) % 10) + 1) for index in range(candidate_count)
        ),
        pad_token_id=tokenizer.pad_token_id,
        retained_logit_count=3,
    )
    utterances = ["xin chào", "xin chào dài"]
    model = DeterministicModel()

    observed = score_candidate_records(model, tokenizer, utterances, plan, device="cpu")

    cached_calls = [call for call in model.call_history if call["has_past"]]
    assert model.calls == 1 + 3
    assert max(call["batch"] for call in cached_calls) == (
        len(utterances) * CANDIDATE_CACHE_CHUNK_SIZE
    )
    assert all(len(scores) == candidate_count for scores in observed)


def test_live_equivalence_guard_rejects_cache_only_divergence():
    class DivergentCachedModel(DeterministicModel):
        def __call__(self, **kwargs):
            cached = kwargs.get("past_key_values") is not None
            outputs = super().__call__(**kwargs)
            if cached:
                outputs.logits[..., 0] += 10.0
            return outputs

    tokenizer = CandidateTokenizer()
    plan = prepare_candidate_scoring_plan(tokenizer, ["alpha", "beta", "gamma"])

    with pytest.raises(RuntimeError, match="diverged from the reference path"):
        validate_candidate_scoring_equivalence(
            DivergentCachedModel(),
            tokenizer,
            "xin chào",
            plan,
            device="cpu",
        )


def test_live_equivalence_guard_rechecks_numeric_drift_in_full_precision(monkeypatch):
    class MixedPrecisionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.low_precision = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.full_precision = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.eval()

    model = MixedPrecisionModel()
    tokenizer = CandidateTokenizer()
    plan = prepare_candidate_scoring_plan(tokenizer, ["alpha", "beta", "gamma"])
    calls = []

    def cached(active_model, _tokenizer, _utterances, active_plan, *, device):
        assert device == "cpu"
        calls.append(("cached", active_model.low_precision.dtype))
        return [{name: 3.0 - index for index, name in enumerate(active_plan.names)}]

    def reference(active_model, _tokenizer, _utterances, active_plan, *, device):
        assert device == "cpu"
        calls.append(("reference", active_model.low_precision.dtype))
        scores = {name: 3.0 - index for index, name in enumerate(active_plan.names)}
        if active_model.low_precision.dtype == torch.bfloat16:
            scores["alpha"] += 0.125
        return [scores]

    monkeypatch.setattr(prompts_module, "score_candidate_records", cached)
    monkeypatch.setattr(prompts_module, "_score_candidate_records_reference", reference)

    assert (
        validate_candidate_scoring_equivalence(
            model,
            tokenizer,
            "xin chào",
            plan,
            device="cpu",
        )
        == 0.0
    )
    assert calls == [
        ("cached", torch.bfloat16),
        ("reference", torch.bfloat16),
        ("cached", torch.float32),
        ("reference", torch.float32),
    ]
    assert model.low_precision.dtype == torch.bfloat16
    assert model.full_precision.dtype == torch.float32


def test_full_precision_recheck_uses_new_tf32_api_and_restores_precision(monkeypatch):
    class LowPrecisionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.eval()

    model = LowPrecisionModel()
    tokenizer = CandidateTokenizer()
    plan = prepare_candidate_scoring_plan(tokenizer, ["alpha", "beta", "gamma"])
    calls = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch,
        "backends",
        SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(fp32_precision="tf32")),
            cudnn=SimpleNamespace(fp32_precision="tf32"),
        ),
    )

    def cached(active_model, _tokenizer, _utterances, active_plan, *, device):
        assert device == "cuda"
        calls.append(
            (
                "cached",
                active_model.weight.dtype,
                torch.backends.cuda.matmul.fp32_precision,
                torch.backends.cudnn.fp32_precision,
            )
        )
        return [{name: 3.0 - index for index, name in enumerate(active_plan.names)}]

    def reference(active_model, _tokenizer, _utterances, active_plan, *, device):
        assert device == "cuda"
        calls.append(
            (
                "reference",
                active_model.weight.dtype,
                torch.backends.cuda.matmul.fp32_precision,
                torch.backends.cudnn.fp32_precision,
            )
        )
        scores = {name: 3.0 - index for index, name in enumerate(active_plan.names)}
        if active_model.weight.dtype == torch.bfloat16:
            scores["alpha"] += 0.125
        return [scores]

    monkeypatch.setattr(prompts_module, "score_candidate_records", cached)
    monkeypatch.setattr(prompts_module, "_score_candidate_records_reference", reference)

    assert (
        validate_candidate_scoring_equivalence(
            model,
            tokenizer,
            "xin chào",
            plan,
            device="cuda",
        )
        == 0.0
    )
    assert calls == [
        ("cached", torch.bfloat16, "tf32", "tf32"),
        ("reference", torch.bfloat16, "tf32", "tf32"),
        ("cached", torch.float32, "ieee", "ieee"),
        ("reference", torch.float32, "ieee", "ieee"),
    ]
    assert model.weight.dtype == torch.bfloat16
    assert torch.backends.cuda.matmul.fp32_precision == "tf32"
    assert torch.backends.cudnn.fp32_precision == "tf32"


def test_full_precision_cuda_controls_fall_back_to_legacy_tf32_api(monkeypatch):
    monkeypatch.delattr(torch.backends.cuda.matmul, "fp32_precision", raising=False)
    monkeypatch.delattr(torch.backends.cudnn, "fp32_precision", raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)

    control, original_matmul, original_cudnn = prompts_module._set_full_precision_cuda_controls(
        torch
    )

    assert control == "allow_tf32"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False

    prompts_module._restore_full_precision_cuda_controls(
        torch,
        control=control,
        original_matmul=original_matmul,
        original_cudnn=original_cudnn,
    )

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


def test_full_precision_recheck_remains_fail_closed_and_restores_dtype(monkeypatch):
    class LowPrecisionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.eval()

    model = LowPrecisionModel()
    tokenizer = CandidateTokenizer()
    plan = prepare_candidate_scoring_plan(tokenizer, ["alpha", "beta", "gamma"])

    def cached(_model, _tokenizer, _utterances, active_plan, *, device):
        assert device == "cpu"
        return [{name: 3.0 - index for index, name in enumerate(active_plan.names)}]

    def divergent_reference(_model, _tokenizer, _utterances, active_plan, *, device):
        assert device == "cpu"
        scores = {name: 3.0 - index for index, name in enumerate(active_plan.names)}
        scores["alpha"] += 0.125
        return [scores]

    monkeypatch.setattr(prompts_module, "score_candidate_records", cached)
    monkeypatch.setattr(
        prompts_module,
        "_score_candidate_records_reference",
        divergent_reference,
    )

    with pytest.raises(RuntimeError, match="precision='full'"):
        validate_candidate_scoring_equivalence(
            model,
            tokenizer,
            "xin chào",
            plan,
            device="cpu",
        )
    assert model.weight.dtype == torch.bfloat16


def test_candidate_scoring_rejects_invalid_inputs_and_retained_logits():
    tokenizer = CandidateTokenizer()
    model = DeterministicModel()
    with pytest.raises(ValueError, match="must not be empty"):
        score_candidates(model, tokenizer, "xin chào", [], device="cpu")
    with pytest.raises(ValueError, match="unique"):
        score_candidates(model, tokenizer, "xin chào", ["alpha", "alpha"], device="cpu")
    with pytest.raises(ValueError, match="normalized"):
        score_candidates(model, tokenizer, "xin chào", [" alpha"], device="cpu")
    plan = prepare_candidate_scoring_plan(tokenizer, ["alpha"])
    with pytest.raises(ValueError, match="at least one utterance"):
        score_candidate_records(model, tokenizer, [], plan, device="cpu")
    invalid_plan = CandidateScoringPlan(
        names=plan.names,
        completions=plan.completions,
        pad_token_id=plan.pad_token_id,
        retained_logit_count=plan.retained_logit_count + 1,
    )
    with pytest.raises(ValueError, match="retained-logit contract"):
        score_candidate_records(model, tokenizer, ["xin chào"], invalid_plan, device="cpu")
    invalid_name_plan = CandidateScoringPlan(
        names=(" alpha",),
        completions=plan.completions,
        pad_token_id=plan.pad_token_id,
        retained_logit_count=plan.retained_logit_count,
    )
    with pytest.raises(ValueError, match="invalid candidate names"):
        score_candidate_records(model, tokenizer, ["xin chào"], invalid_name_plan, device="cpu")

    class TrainingModel(DeterministicModel):
        training = True

    with pytest.raises(RuntimeError, match="evaluation mode"):
        score_candidate_records(TrainingModel(), tokenizer, ["xin chào"], plan, device="cpu")

    class FullLogitModel(DeterministicModel):
        def __call__(self, **kwargs):
            kwargs["logits_to_keep"] = 0
            return super().__call__(**kwargs)

    with pytest.raises(RuntimeError, match="retained-logit shape"):
        score_candidate_records(FullLogitModel(), tokenizer, ["xin chào"], plan, device="cpu")

    class NonFiniteModel(DeterministicModel):
        def __call__(self, **kwargs):
            outputs = super().__call__(**kwargs)
            outputs.logits.fill_(float("nan"))
            return outputs

    with pytest.raises(RuntimeError, match="non-finite"):
        score_candidate_records(NonFiniteModel(), tokenizer, ["xin chào"], plan, device="cpu")


def test_cached_record_scoring_matches_tiny_qwen_scalar_path():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(20260714)
    model = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )
    model.eval()
    tokenizer = CandidateTokenizer()
    candidates = ["alpha", "beta", "gamma"]
    utterances = ["xin chào", "xin chào dài"]
    plan = prepare_candidate_scoring_plan(tokenizer, candidates)

    observed = score_candidate_records(model, tokenizer, utterances, plan, device="cpu")
    expected = []
    with torch.inference_mode():
        for utterance in utterances:
            prompt = prompt_token_ids(tokenizer, utterance)
            scores = {}
            for candidate in candidates:
                completion = label_token_ids(tokenizer, candidate)
                values = torch.tensor([[*prompt, *completion]], dtype=torch.long)
                logits = (
                    model(
                        input_ids=values,
                        attention_mask=torch.ones_like(values),
                        use_cache=False,
                    )
                    .logits[0]
                    .numpy()
                )
                scores[candidate] = conditional_token_mean_log_likelihood(
                    logits,
                    prompt_length=len(prompt),
                    candidate_token_ids=completion,
                )
            expected.append(scores)

    for observed_scores, expected_scores in zip(observed, expected, strict=True):
        assert observed_scores == pytest.approx(expected_scores, rel=2e-5, abs=2e-5)


def test_record_scoring_matches_pinned_qwen2_scalar_path():
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    torch.manual_seed(20260714)
    base = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.eval()
    tokenizer = CandidateTokenizer()
    candidates = ["alpha", "beta", "gamma"]
    utterances = ["xin chào", "xin chào dài"]
    plan = prepare_candidate_scoring_plan(tokenizer, candidates)

    observed = score_candidate_records(model, tokenizer, utterances, plan, device="cpu")
    expected = []
    with torch.inference_mode():
        for utterance in utterances:
            prompt = prompt_token_ids(tokenizer, utterance)
            scores = {}
            for candidate in candidates:
                completion = label_token_ids(tokenizer, candidate)
                values = torch.tensor([[*prompt, *completion]], dtype=torch.long)
                logits = (
                    model(
                        input_ids=values,
                        attention_mask=torch.ones_like(values),
                        use_cache=False,
                    )
                    .logits[0]
                    .numpy()
                )
                scores[candidate] = conditional_token_mean_log_likelihood(
                    logits,
                    prompt_length=len(prompt),
                    candidate_token_ids=completion,
                )
            expected.append(scores)

    for observed_scores, expected_scores in zip(observed, expected, strict=True):
        assert observed_scores == pytest.approx(expected_scores, rel=2e-5, abs=2e-5)
