import tempfile
import unittest
from pathlib import Path

from src.sd15.prepare_runtime_cache import prepare_runtime_cache


class RuntimeCacheTest(unittest.TestCase):
    def test_materializes_snapshot_without_changing_blobs(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "refs").mkdir()
            (cache / "refs/main").write_text("revision\n")
            blob = cache / "blobs/config"
            blob.parent.mkdir()
            contents = b'{"test": true}\n'
            blob.write_bytes(contents)
            snapshot = cache / "snapshots/revision"
            config = snapshot / "scheduler/config.json"
            config.parent.mkdir(parents=True)
            config.symlink_to("../../../blobs/config")
            regular = snapshot / "regular.json"
            regular.write_bytes(contents)

            self.assertEqual(prepare_runtime_cache(cache), 1)
            self.assertFalse(config.is_symlink())
            self.assertEqual(config.read_bytes(), contents)
            self.assertEqual(blob.read_bytes(), contents)
            self.assertEqual(regular.read_bytes(), contents)
            self.assertEqual(prepare_runtime_cache(cache), 0)

    def test_missing_blob_preserves_link(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "refs").mkdir()
            (cache / "refs/main").write_text("revision")
            snapshot = cache / "snapshots/revision"
            snapshot.mkdir(parents=True)
            link = snapshot / "missing.json"
            link.symlink_to("../../blobs/missing")

            with self.assertRaises(FileNotFoundError):
                prepare_runtime_cache(cache)
            self.assertTrue(link.is_symlink())
            self.assertEqual(list(snapshot.iterdir()), [link])
