"""Light stub of chromadb (tests only)."""
from types import ModuleType
import sys
class PersistentClient:  # minimal
    def get_collection(self,*a,**k): return None
    def create_collection(self,*a,**k): return None
    def list_collections(self,*a,**k): return []

api=ModuleType("chromadb.api")
conf=ModuleType("chromadb.api.configuration")
class CollectionConfigurationInternal:  # noqa
    @staticmethod
    def from_json(*a,**k): return CollectionConfigurationInternal()
    @staticmethod
    def from_json_str(*a,**k): return CollectionConfigurationInternal()
conf.CollectionConfigurationInternal=CollectionConfigurationInternal  # type: ignore
api.configuration=conf  # type: ignore
sys.modules["chromadb.api"]=api
sys.modules["chromadb.api.configuration"]=conf
__all__=["PersistentClient","api"]