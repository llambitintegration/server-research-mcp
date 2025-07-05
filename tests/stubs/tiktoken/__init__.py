# Stubbed tiktoken package (moved to tests/stubs)

class _Encoding:
    def encode(self, text, *_, **__):
        return list(str(text).encode())
    def decode(self, tokens, *_, **__):
        return bytes(tokens).decode(errors="ignore")

def get_encoding(name:str):
    return _Encoding()

def encoding_for_model(model:str):
    return _Encoding()

__all__=["get_encoding","encoding_for_model"]

import types,sys
_registry=types.ModuleType("tiktoken.registry")
_registry.get_encoding=get_encoding  # type: ignore
sys.modules["tiktoken.registry"]=_registry