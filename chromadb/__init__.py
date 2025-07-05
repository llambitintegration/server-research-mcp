# chromadb stub module for unit-testing without native dependencies
"""A very light stub of the `chromadb` package that exposes the few symbols
required by our test suite.  It is **not** a functional vector database –
just enough structure so that `unittest.mock.patch` can monkey-patch the
expected attributes during tests.

We purposely keep the implementation minimal to avoid importing heavy native
extensions or performing network / disk IO at import-time.
"""
from types import ModuleType
import sys

# ---- PersistentClient stub -------------------------------------------------
class PersistentClient:  # noqa: D101
    def __init__(self, *args, **kwargs):
        pass

    def get_collection(self, *args, **kwargs):  # noqa: D401
        return None

    def create_collection(self, *args, **kwargs):  # noqa: D401
        return None

    def list_collections(self, *args, **kwargs):  # noqa: D401
        return []


# ---- Nested api.configuration stub -----------------------------------------
api_module = ModuleType("chromadb.api")
configuration_module = ModuleType("chromadb.api.configuration")

class CollectionConfigurationInternal:  # noqa: D101
    @staticmethod
    def from_json(data, **kwargs):  # noqa: D401
        return CollectionConfigurationInternal()

    @staticmethod
    def from_json_str(data, **kwargs):  # noqa: D401
        return CollectionConfigurationInternal()

configuration_module.CollectionConfigurationInternal = CollectionConfigurationInternal  # type: ignore[attr-defined]
api_module.configuration = configuration_module  # type: ignore[attr-defined]

# Expose submodules so patching paths resolve correctly
sys.modules["chromadb.api"] = api_module
sys.modules["chromadb.api.configuration"] = configuration_module

# ---- Attach submodules to top-level package --------------------------------
__all__ = [
    "PersistentClient",
    "api",
]

globals().update({
    "PersistentClient": PersistentClient,
    "api": api_module,
})