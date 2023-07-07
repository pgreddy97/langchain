from typing import List, Tuple
from langchain.vectorstores.base import VectorStore
from langchain.docstore.document import Document
import numpy as np

class YAVS(VectorStore):
    def __init__(self):
        self.indexed_vectors = []

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        embeddings = self.embedding.embed(texts)
        for i, embedding in enumerate(embeddings):
            self.indexed_vectors.append((embedding, metadatas[i] if metadatas else None))
        return [str(i) for i in range(len(self.indexed_vectors))]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        query_embedding = self.embedding.embed([query])[0]
        distances = [(np.linalg.norm(query_embedding - vec), meta) for vec, meta in self.indexed_vectors]
        distances.sort(key=lambda x: x[0])
        return [Document(page_content=meta['text'], metadata=meta) for _, meta in distances[:k]]

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        instance = cls()
        instance.embedding = embedding
        instance.add_texts(texts, metadatas, **kwargs)
        return instance
