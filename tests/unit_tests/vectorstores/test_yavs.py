import numpy as np
from langchain.vectorstores.yavs import YAVS
from langchain.docstore.document import Document
from typing import List, Optional
import unittest

class TestYAVS(unittest.TestCase):
    def setUp(self):
        self.yavs = YAVS()
        self.texts = ["This is a test", "Another test", "Yet another test", "Final test"]
        self.metadatas = [{'text': text} for text in self.texts]

    def test_add_texts(self):
        ids = self.yavs.add_texts(self.texts, self.metadatas)
        self.assertEqual(len(ids), len(self.texts))
        self.assertEqual(self.yavs.indexed_vectors[0][1]['text'], self.texts[0])

    def test_similarity_search(self):
        self.yavs.add_texts(self.texts, self.metadatas)
        results = self.yavs.similarity_search("test", k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], Document)

    def test_from_texts(self):
        yavs = YAVS.from_texts(self.texts, self.yavs.embedding, self.metadatas)
        self.assertEqual(len(yavs.indexed_vectors), len(self.texts))

if __name__ == "__main__":
    unittest.main()
