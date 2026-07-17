"""Tests for grpo_math.eval.backends: Completion, GenerationBackend protocol, FakeBackend."""

from grpo_math.eval.backends import Completion, FakeBackend


def test_completion_fields():
    c = Completion(text="hello world", finish_reason="stop", n_tokens=2)
    assert c.text == "hello world"
    assert c.finish_reason == "stop"
    assert c.n_tokens == 2


def test_fake_backend_dict_script_basic():
    backend = FakeBackend({"p1": ["a b c", "d e"]})
    out = backend.generate(["p1"], k=2, temperature=0.6, top_p=0.95, max_tokens=100, seed=0)
    assert len(out) == 1
    assert len(out[0]) == 2
    assert out[0][0].text == "a b c"
    assert out[0][0].n_tokens == 3
    assert out[0][0].finish_reason == "stop"
    assert out[0][1].text == "d e"
    assert out[0][1].n_tokens == 2


def test_fake_backend_truncation_sentinel_strips_and_flags():
    backend = FakeBackend({"p1": ["short answer<TRUNCATED>"]})
    out = backend.generate(["p1"], k=1, temperature=0.6, top_p=0.95, max_tokens=100, seed=0)
    completion = out[0][0]
    assert completion.finish_reason == "length"
    assert "<TRUNCATED>" not in completion.text
    assert completion.text == "short answer"


def test_fake_backend_non_truncated_is_stop():
    backend = FakeBackend({"p1": ["a normal answer"]})
    out = backend.generate(["p1"], k=1, temperature=0.6, top_p=0.95, max_tokens=100, seed=0)
    assert out[0][0].finish_reason == "stop"


def test_fake_backend_callable_script():
    def script(prompt, i):
        return f"{prompt}-{i}"

    backend = FakeBackend(script)
    out = backend.generate(["p1", "p2"], k=2, temperature=0.6, top_p=0.95, max_tokens=100, seed=0)
    assert out[0][0].text == "p1-0"
    assert out[0][1].text == "p1-1"
    assert out[1][0].text == "p2-0"
    assert out[1][1].text == "p2-1"


def test_fake_backend_multiple_prompts_alignment():
    backend = FakeBackend({"p1": ["a"], "p2": ["b"]})
    out = backend.generate(["p1", "p2"], k=1, temperature=0.6, top_p=0.95, max_tokens=10, seed=0)
    assert out[0][0].text == "a"
    assert out[1][0].text == "b"


def test_fake_backend_cycles_when_script_shorter_than_k():
    backend = FakeBackend({"p1": ["x", "y"]})
    out = backend.generate(["p1"], k=5, temperature=0.6, top_p=0.95, max_tokens=10, seed=0)
    texts = [c.text for c in out[0]]
    assert texts == ["x", "y", "x", "y", "x"]


def test_fake_backend_deterministic():
    backend = FakeBackend({"p1": ["x y z"]})
    out1 = backend.generate(["p1"], k=1, temperature=0.6, top_p=0.95, max_tokens=10, seed=0)
    out2 = backend.generate(["p1"], k=1, temperature=0.6, top_p=0.95, max_tokens=10, seed=0)
    assert out1[0][0].text == out2[0][0].text
    assert out1[0][0].finish_reason == out2[0][0].finish_reason


def test_import_backends_without_vllm_or_datasets():
    # module-level import must never require vllm; this test is meaningful
    # because vllm is genuinely absent from this venv.
    import grpo_math.eval.backends  # noqa: F401
