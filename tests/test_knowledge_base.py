import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from campusmind import knowledge_base as kb


class IndexManifestTests(unittest.TestCase):
    def write_record(self, root: Path, record: dict) -> None:
        folder = root / "campus_test"
        folder.mkdir(exist_ok=True)
        (folder / "index_manifest.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )

    def test_load_index_manifest(self):
        record = {
            "collection_name": "campus_test", "state": "ready",
            "chunk_count": 3, "parent_count": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_record(root, record)
            with patch.object(kb, "INDEX_ROOT", root):
                loaded = kb.load_index_manifest("campus_test")
                self.assertEqual(loaded["chunk_count"], 3)
                self.assertEqual(loaded["parent_count"], 2)

                invalid_records = (
                    {**record, "state": "building"},
                    {**record, "chunk_count": 0},
                    {**record, "parent_count": "2"},
                    {**record, "collection_name": "other_collection"},
                )
                for invalid in invalid_records:
                    with self.subTest(record=invalid):
                        self.write_record(root, invalid)
                        with self.assertRaises(ValueError):
                            kb.load_index_manifest("campus_test")

    def test_invalid_collection_name(self):
        with self.assertRaises(ValueError):
            kb.load_index_manifest("bad/name")


if __name__ == "__main__":
    unittest.main()
