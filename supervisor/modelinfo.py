"""What the model actually is, from Ollama: name, size, parameters,
quantization, context window, capabilities, memory use. For the Stats tab.

Uses /api/show, /api/tags, /api/ps and /api/version. Each is optional: an
older Ollama, or one that's unreachable, just leaves those fields empty.
"""

from __future__ import annotations

import json
import time
import urllib.request

_CACHE: dict = {}
CACHE_SECONDS = 20


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def parse_parameters(text: str) -> dict:
    """The Modelfile PARAMETER lines ("num_ctx 49152\\nstop \\"<|im_end|>\\"")
    as a dict; repeated keys (like stop) become lists."""
    out: dict = {}
    for line in (text or "").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts[0], parts[1].strip().strip('"')
        try:
            value = int(value) if value.lstrip("-").isdigit() else float(value) if "." in value else value
        except ValueError:
            pass
        if key in out:
            out[key] = (out[key] if isinstance(out[key], list) else [out[key]]) + [value]
        else:
            out[key] = value
    return out


def _same_model(a: str, b: str) -> bool:
    norm = lambda n: n if ":" in n else n + ":latest"
    return norm(a or "") == norm(b or "")


def summarize(model: str, show: dict | None, tags: dict | None, ps: dict | None, version: dict | None,
              settings) -> dict:
    show = show or {}
    details = show.get("details") or {}
    info = show.get("model_info") or {}
    arch = info.get("general.architecture") or details.get("family") or ""
    params = parse_parameters(show.get("parameters", ""))
    tag = next((m for m in (tags or {}).get("models", []) if _same_model(m.get("name", ""), model)), {})
    loaded = next((m for m in (ps or {}).get("models", []) if _same_model(m.get("name", ""), model)), None)

    model_max = next((v for k, v in info.items() if k.endswith(".context_length")), None)
    num_ctx = params.get("num_ctx")
    loaded_ctx = (loaded or {}).get("context_length")  # newer Ollama reports what it loaded with
    effective = loaded_ctx or num_ctx
    needed = settings.max_input_tokens + settings.max_output_tokens
    if effective:
        ok = effective >= needed
        context_note = (f"OK: {effective:,} covers the up to {needed:,} tokens the supervisor uses "
                        f"(max_input_tokens {settings.max_input_tokens:,} + max_output_tokens {settings.max_output_tokens:,})."
                        if ok else
                        f"Too small: Ollama's context is {effective:,} tokens, but the supervisor sends up to "
                        f"{settings.max_input_tokens:,} tokens and asks for up to {settings.max_output_tokens:,} more. "
                        f"Ollama silently cuts the start of long prompts, so the agent forgets its task. Raise num_ctx "
                        f"on the model (or OLLAMA_CONTEXT_LENGTH) to at least {needed:,}, or lower max_input_tokens.")
    else:
        ok = None
        context_note = ("num_ctx isn't set on this model, so Ollama's default (or OLLAMA_CONTEXT_LENGTH on the "
                        f"Ollama machine) applies. Make sure it's at least {needed:,}, or long prompts get cut.")

    beyond = bool(effective and model_max and effective > model_max)
    if beyond:
        context_note += (f" Note: that's more than the {model_max:,} tokens the model was trained on, so answers can get "
                         "worse once a conversation grows past that.")

    vram, size_mem = (loaded or {}).get("size_vram"), (loaded or {}).get("size")
    return {
        "name": model,
        "reachable": bool(show or tags or version),
        "ollama_url": settings.ollama_url,
        "ollama_version": (version or {}).get("version"),
        "architecture": arch,
        "family": details.get("family"),
        "parent_model": details.get("parent_model") or None,
        "format": details.get("format"),
        "parameter_size": details.get("parameter_size"),
        "parameter_count": info.get("general.parameter_count"),
        "quantization": details.get("quantization_level"),
        "size_on_disk": tag.get("size"),
        "digest": (tag.get("digest") or "")[:12] or None,
        "modified_at": tag.get("modified_at") or show.get("modified_at"),
        "capabilities": show.get("capabilities") or [],
        "context": {
            "model_max": model_max,
            "num_ctx": num_ctx,
            "loaded": loaded_ctx,
            "needed": needed,
            "ok": ok,
            "beyond_training": beyond,
            "note": context_note,
        },
        "loaded": None if loaded is None else {
            "size": size_mem,
            "vram": vram,
            "gpu_percent": round(100 * vram / size_mem) if vram and size_mem else 0 if size_mem else None,
            "expires_at": loaded.get("expires_at"),
        },
        "layers": next((v for k, v in info.items() if k.endswith(".block_count")), None),
        "embedding": next((v for k, v in info.items() if k.endswith(".embedding_length")), None),
        "model_defaults": {k: params[k] for k in ("temperature", "top_p", "top_k", "repeat_penalty", "min_p",
                                                   "presence_penalty", "num_predict") if k in params},
        "supervisor_sampling": {k: v for k, v in (
            ("temperature", settings.temperature), ("top_p", settings.top_p), ("top_k", settings.top_k),
            ("presence_penalty", settings.presence_penalty or None), ("max_output_tokens", settings.max_output_tokens),
        ) if v is not None},
    }


def fetch(settings, timeout: float = 6, use_cache: bool = True) -> dict:
    key = (settings.ollama_url, settings.model)
    hit = _CACHE.get(key)
    if use_cache and hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    base = settings.ollama_url.rstrip("/")
    errors = []

    def attempt(fn, *args):
        try:
            return fn(*args, timeout)
        except Exception as exc:
            errors.append(f"{args[0].rsplit('/', 1)[-1]}: {exc}")
            return None

    # The cheap call first: if Ollama can't be reached, don't wait on three more timeouts.
    version = attempt(_get, base + "/api/version")
    if version is None:
        show = tags = ps = None
    else:
        show = attempt(_post, base + "/api/show", {"model": settings.model})
        tags = attempt(_get, base + "/api/tags")
        ps = attempt(_get, base + "/api/ps")
    data = summarize(settings.model, show, tags, ps, version, settings)
    data["errors"] = errors
    _CACHE[key] = (time.monotonic(), data)
    return data
