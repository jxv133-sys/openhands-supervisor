from supervisor.config import Settings
from supervisor.modelinfo import parse_parameters, summarize

SHOW = {
    "parameters": 'num_ctx                        49152\nstop                           "<|im_start|>"\nstop                           "<|im_end|>"\ntemperature                    0.6\ntop_k                          20',
    "details": {"parent_model": "", "format": "gguf", "family": "qwen2", "parameter_size": "32.8B", "quantization_level": "Q4_K_M"},
    "model_info": {"general.architecture": "qwen2", "general.parameter_count": 32763876352,
                   "qwen2.context_length": 32768, "qwen2.block_count": 64, "qwen2.embedding_length": 5120},
    "capabilities": ["completion", "tools", "thinking"],
}
TAGS = {"models": [{"name": "qwen-abl:latest", "size": 19851349856, "digest": "abcdef1234567890", "modified_at": "2026-10-01T10:00:00Z"}]}
PS = {"models": [{"name": "qwen-abl:latest", "size": 25000000000, "size_vram": 22500000000, "expires_at": "2026-11-24T16:00:00Z"}]}


def test_parse_parameters():
    p = parse_parameters(SHOW["parameters"])
    assert p["num_ctx"] == 49152 and p["temperature"] == 0.6 and p["stop"] == ["<|im_start|>", "<|im_end|>"]


def test_summarize_full():
    s = Settings()
    m = summarize("qwen-abl:latest", SHOW, TAGS, PS, {"version": "0.12.3"}, s)
    assert m["parameter_size"] == "32.8B" and m["quantization"] == "Q4_K_M" and m["architecture"] == "qwen2"
    assert m["size_on_disk"] == 19851349856 and m["loaded"]["gpu_percent"] == 90
    assert m["context"]["num_ctx"] == 49152 and m["context"]["ok"] is True and m["context"]["model_max"] == 32768
    assert m["context"]["beyond_training"] is True and "trained on" in m["context"]["note"]
    assert m["capabilities"] == ["completion", "tools", "thinking"] and m["ollama_version"] == "0.12.3"
    assert m["model_defaults"] == {"temperature": 0.6, "top_k": 20}


def test_context_too_small_and_unknown():
    s = Settings()
    small = dict(SHOW, parameters="num_ctx 8192")
    m = summarize("qwen-abl", small, TAGS, None, None, s)  # "qwen-abl" matches "qwen-abl:latest"
    assert m["context"]["ok"] is False and "cuts the start" in m["context"]["note"] and m["size_on_disk"]
    m = summarize("qwen-abl", dict(SHOW, parameters=""), None, None, None, s)
    assert m["context"]["ok"] is None and m["loaded"] is None


def test_unreachable_ollama():
    m = summarize("x", None, None, None, None, Settings())
    assert m["reachable"] is False and m["parameter_size"] is None


def test_fetch_gives_up_fast_when_ollama_is_down():
    import time

    from supervisor import modelinfo

    s = Settings(); s.ollama_url = "http://127.0.0.1:9"  # nothing listens here
    start = time.monotonic()
    m = modelinfo.fetch(s, timeout=2, use_cache=False)
    assert m["reachable"] is False and len(m["errors"]) == 1 and time.monotonic() - start < 3
