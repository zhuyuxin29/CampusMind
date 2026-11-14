import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from campusmind import knowledge_base as kb, query_processing as q, reranking as r, retrieval as ret
from campusmind import generation as g

ROOT = Path(__file__).resolve().parents[1]


def parent(key, text, start=1, end=2):
    return Document(page_content=text, metadata={"parent_id": key, "source": "test.pdf",
        "section_id": "section", "scholarship_type": "国家奖学金", "section_title": "测试制度",
        "page_start": start, "page_end": end, "page_number": start,
        "page_spans": [{"page_number": start, "start": 0, "end": len(text)}]})


def child(key, score=0.9):
    return Document(page_content="子片段", metadata={"parent_id": key, "source": "test.pdf",
        "section_id": "section", "scholarship_type": "国家奖学金", "page_number": 2,
        "rerank_score": score, "rerank_rank": 1, "chunk_id": key + "-child"})


class ModuleTests(unittest.TestCase):
    def test_display_range(self):
        self.assertIn("测试制度｜PDF 第 1–2 页", ret.format_document_source(parent("a", "正文")))

    def test_v2_chunk_id_without_legacy_index(self):
        doc = Document(page_content="正文", metadata={"split_version": kb.SPLIT_VERSION, "chunk_id": "a" * 64})
        self.assertEqual(ret._get_chunk_id(doc), "a" * 64)
        self.assertEqual(len(ret.fuse_rankings([[doc, doc], [doc]])), 1)

    def test_fuse_rankings_dedup_fusion_and_metadata(self):
        first = Document(page_content="甲", metadata={"chunk_id": "a" * 64,
                                                      "split_version": kb.SPLIT_VERSION})
        second = Document(page_content="乙", metadata={"chunk_id": "b" * 64,
                                                       "split_version": kb.SPLIT_VERSION})
        fused = ret.fuse_rankings([[first, second], [second, first]], top_k=2)
        self.assertEqual([d.metadata["chunk_id"] for d in fused], ["a" * 64, "b" * 64])
        self.assertEqual([d.metadata["retrieval_rank"] for d in fused], [1, 2])
        self.assertAlmostEqual(fused[0].metadata["rrf_score"], fused[1].metadata["rrf_score"])
        self.assertIsNot(fused[0], first)
        self.assertNotIn("rrf_score", first.metadata)

    def test_fuse_rankings_limits_and_validation(self):
        docs = [Document(page_content=str(i), metadata={"chunk_id": f"{i:064x}",
                                                        "split_version": kb.SPLIT_VERSION})
                for i in range(3)]
        self.assertEqual(len(ret.fuse_rankings([docs], top_k=2)), 2)
        self.assertEqual(ret.fuse_rankings([]), [])
        for kwargs in ({"top_k": 0}, {"rrf_constant": 0}):
            with self.assertRaises(ValueError):
                ret.fuse_rankings([docs], **kwargs)

    def test_open_index_validates_contract(self):
        parents = {"parent": parent("parent", "父级原文")}
        store = Mock()
        store.get.return_value = {
            "ids": ["child"], "documents": ["子片段"],
            "metadatas": [{"split_version": kb.SPLIT_VERSION, "parent_id": "parent"}],
        }
        with patch.object(ret, "load_index_manifest",
                          return_value={"chunk_count": 1, "parent_count": 1}), \
             patch.object(ret, "load_parent_documents", return_value=parents), \
             patch.object(ret, "open_vector_store", return_value=store), \
             patch.object(ret, "build_bm25_retriever"), \
             patch.object(ret, "build_vector_retriever"), \
             patch.object(ret, "HybridRetriever") as hybrid:
            retriever, loaded_parents = ret.open_index("campus_test")
        self.assertIs(loaded_parents, parents)
        self.assertIs(retriever, hybrid.return_value)

    def test_open_index_rejects_chunk_count_mismatch(self):
        parents = {"parent": parent("parent", "父级原文")}
        store = Mock()
        store.get.return_value = {
            "ids": ["child"], "documents": ["子片段"],
            "metadatas": [{"split_version": kb.SPLIT_VERSION, "parent_id": "parent"}],
        }
        with patch.object(ret, "load_index_manifest",
                          return_value={"chunk_count": 2, "parent_count": 1}), \
             patch.object(ret, "load_parent_documents", return_value=parents), \
             patch.object(ret, "open_vector_store", return_value=store), \
             patch.object(ret, "build_bm25_retriever"), \
             patch.object(ret, "build_vector_retriever"), \
             patch.object(ret, "HybridRetriever"):
            with self.assertRaisesRegex(ValueError, "片段数"):
                ret.open_index("campus_test")

    def test_open_index_rejects_mismatched_chunks(self):
        store = Mock()
        store.get.return_value = {
            "ids": ["child"], "documents": ["子片段"],
            "metadatas": [{"split_version": kb.SPLIT_VERSION, "parent_id": "missing"}],
        }
        with patch.object(ret, "load_index_manifest",
                          return_value={"chunk_count": 1, "parent_count": 0}), \
             patch.object(ret, "load_parent_documents", return_value={}), \
             patch.object(ret, "open_vector_store", return_value=store):
            with self.assertRaisesRegex(ValueError, "父级"):
                ret.open_index("campus_test")
        store.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        with patch.object(ret, "load_index_manifest",
                          return_value={"chunk_count": 1, "parent_count": 0}), \
             patch.object(ret, "load_parent_documents", return_value={}), \
             patch.object(ret, "open_vector_store", return_value=store):
            with self.assertRaisesRegex(ValueError, "为空"):
                ret.open_index("campus_test")
