# tiktoken stub for testing purposes without native binaries

class _Encoding:
    """Minimal encoding class that mimics the interface expected by OpenAI/langchain."""

    def encode(self, text, *args, **kwargs):
        # Very naive: return list of bytes so len() gives token count surrogate
        if isinstance(text, str):
            return list(text.encode("utf-8"))
        # Fallback for other types
        return list(str(text).encode("utf-8"))

    def decode(self, tokens, *args, **kwargs):
        # Inverse of encode – assume tokens are ints representing byte values
        try:
            return bytes(tokens).decode("utf-8", errors="ignore")
        except Exception:
            return ""


def get_encoding(name: str):  # noqa: D401
    """Return a dummy Encoding instance regardless of name."""
    return _Encoding()


def encoding_for_model(model: str):  # noqa: D401
    """Return a dummy Encoding instance for a given model string."""
    return _Encoding()


# Exported names expected by callers
__all__ = [
    "get_encoding",
    "encoding_for_model",
]

# Optional registry submodule used by some libraries
import types, sys  # noqa: E402
registry_module = types.ModuleType("tiktoken.registry")
registry_module.get_encoding = get_encoding  # type: ignore[attr-defined]
sys.modules["tiktoken.registry"] = registry_module