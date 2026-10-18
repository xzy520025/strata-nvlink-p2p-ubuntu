"""Failure/transaction tests for the P2P importer; no model or GPU needed."""
from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).parent / "p2p/installer.py"
if not SOURCE.is_file():
    SOURCE = Path(__file__).resolve().parents[1] / "installer.py"
spec = importlib.util.spec_from_file_location("p2p_installer_tested", SOURCE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class ImporterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p2p 测试 空格-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "serve").mkdir()
        (self.root / "serve/server.py").write_text("# --peer-device\ndef engine_args(): pass\ndef layer_split_of(): pass\n")
        self.profile = self.root / "模型 profile.bin"; self.profile.write_bytes(b"fixture")
        self.cfg = {"exe": "/old/strata", "args": ["--expert-profile", self.profile.name,
            "--expert-cache", "auto", "--resident-budget-gib", "40", "--layer-split", "auto",
            "--trim-stage-weights", "--ctx", "262144"], "cwd": str(self.root), "gpu": [1, 0],
            "host": "0.0.0.0", "api_key": "private-fixture-key", "port": 8282,
            "parallel": 2, "vision": {"exe": "/old/vision"}, "env": {"KEEP": "yes"},
            "layer_split": "auto", "split_skip_if_fits": True, "lib_dirs": ["/old/cuda"]}
        self.config = self.root / "配置 文件.json"
        self.original = json.dumps(self.cfg, ensure_ascii=False, indent=1).encode()
        self.config.write_bytes(self.original)
        self.options = SimpleNamespace(expert_profile=None, resident_budget_gib=None, gpus=None)
        self.engine = self.root / "engine/p2p-test/strata"

    def prepared(self, cfg=None):
        return m.prepare_config(self.cfg if cfg is None else cfg, self.root, self.engine, self.options)

    def test_convert_layer_split_preserves_private_and_model_settings(self):
        new = self.prepared()
        for k in ("api_key", "host", "port", "parallel", "vision", "cwd"):
            self.assertEqual(new[k], self.cfg[k])
        self.assertEqual(new["gpu"], [1, 0])
        self.assertEqual(new["env"]["KEEP"], "yes")
        self.assertEqual(m.flag_value(new["args"], "--peer-device"), "1")
        self.assertEqual(m.flag_value(new["args"], "--peer-slots"), "0")
        self.assertNotIn("--layer-split", new["args"])
        self.assertNotIn("--trim-stage-weights", new["args"])
        self.assertEqual(m.flag_value(new["args"], "--ctx"), "262144")
        self.assertEqual(self.config.read_bytes(), self.original)

    def test_idempotent_peer_conversion_and_explicit_cache_settings(self):
        new = self.prepared()
        new["args"] = m.without_flags(new["args"], {"--peer-reserve-mib", "--peer-slots"}) + ["--peer-reserve-mib", "1700", "--peer-slots", "4000"]
        first = self.prepared(new); second = self.prepared(first)
        self.assertEqual(first, second)
        self.assertEqual(m.flag_value(first["args"], "--peer-slots"), "4000")
        for flag in m.PEER_SWITCHES:
            self.assertEqual(second["args"].count(flag), 1)

    def test_remove_conflicting_remote_helpers_and_equals_syntax(self):
        self.cfg["args"] += ["--expert-cache-device1=auto", "--remote-expert-opt", "--expert-cache-remote-placement", "layer"]
        new = self.prepared()
        self.assertFalse(any("expert-cache-device1" in s or "remote-expert-opt" in s for s in new["args"]))
        self.assertNotIn("--expert-cache-remote-placement", new["args"])
        self.assertNotIn("layer", new["args"])

    def test_missing_profile_does_not_touch_config(self):
        self.profile.unlink()
        with self.assertRaisesRegex(m.InstallError, "profile"):
            self.prepared()
        self.assertEqual(self.config.read_bytes(), self.original)

    def test_missing_budget_requires_explicit_target_budget(self):
        self.cfg["args"] = m.without_flags(self.cfg["args"], {"--resident-budget-gib"})
        with self.assertRaisesRegex(m.InstallError, "RAM"):
            self.prepared()
        self.options.resident_budget_gib = 24
        self.assertEqual(m.flag_value(self.prepared()["args"], "--resident-budget-gib"), "24")

    def test_invalid_budget_and_disabled_cache_are_rejected(self):
        for value in (float("nan"), float("inf"), 0, -1):
            self.options.resident_budget_gib = value
            with self.subTest(value=value), self.assertRaises(m.InstallError): self.prepared()
        self.options.resident_budget_gib = None
        self.cfg["args"] = m.without_flags(self.cfg["args"], {"--expert-cache"}) + ["--expert-cache", "0"]
        with self.assertRaises(m.InstallError): self.prepared()

    def test_duplicate_value_or_malformed_args_rejected(self):
        self.cfg["args"] += ["--expert-profile", str(self.profile)]
        with self.assertRaisesRegex(m.InstallError, "重复"):
            self.prepared()
        with self.assertRaises(m.InstallError):
            m.without_flags(["--layer-split"], {"--layer-split"})

    def test_lan_key_requirement_and_string_args_validation(self):
        self.cfg["api_key"] = ""
        with self.assertRaisesRegex(m.InstallError, "API key"):
            self.prepared()
        self.cfg["host"] = "127.0.0.1"
        self.prepared()
        self.cfg["args"] = "--expert-cache auto"
        with self.assertRaises(m.InstallError): self.prepared()

    def test_gpu_order_and_invalid_selection(self):
        self.assertEqual(m.select_gpus(self.cfg, "0,1"), [0, 1])
        self.assertEqual(m.select_gpus({"gpu": 0}, None), [0, 1])
        for raw in ("0,0", "0", "x,1", "-1,0", "0,1,2"):
            with self.assertRaises(m.InstallError): m.select_gpus(self.cfg, raw)

    def test_backup_permissions_rollback_bytes_and_idempotence(self):
        new = self.prepared()
        backup = m.commit_config(self.config, self.original, new, self.root / "backups", self.root)
        self.assertEqual(stat.S_IMODE(backup.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((backup / "config.before.json").stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.config.stat().st_mode), 0o600)
        installed = self.config.read_bytes()
        self.assertIsNone(m.commit_config(self.config, installed, new, self.root / "backups", self.root))
        with contextlib.redirect_stdout(io.StringIO()):
            m.rollback(backup); m.rollback(backup)
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertEqual(len(list((self.root / "backups").iterdir())), 1)

    def test_concurrent_config_change_stops_commit(self):
        self.config.write_bytes(b'{"edited":true}')
        with self.assertRaisesRegex(m.InstallError, "修改"):
            m.commit_config(self.config, self.original, self.prepared(), self.root / "backups", self.root)
        self.assertEqual(self.config.read_bytes(), b'{"edited":true}')

    def test_failed_atomic_replace_leaves_original_config(self):
        replace = m.os.replace
        def denied(src, dst):
            if Path(dst) == self.config:
                raise PermissionError('fixture: cannot replace config')
            return replace(src, dst)
        with patch.object(m.os, 'replace', side_effect=denied):
            with self.assertRaises(PermissionError):
                m.commit_config(self.config, self.original, self.prepared(), self.root / 'backups', self.root)
        self.assertEqual(self.config.read_bytes(), self.original)
        backup = next((self.root / 'backups').iterdir())
        self.assertEqual((backup / 'config.before.json').read_bytes(), self.original)

    def test_rollback_preserves_later_edits_and_rejects_bad_backup(self):
        backup = m.commit_config(self.config, self.original, self.prepared(), self.root / "backups", self.root)
        self.config.write_bytes(b'{"user-edit":true}')
        with self.assertRaises(m.InstallError): m.rollback(backup)
        self.assertEqual(self.config.read_bytes(), b'{"user-edit":true}')
        (backup / "config.before.json").write_bytes(b"damaged")
        with self.assertRaisesRegex(m.InstallError, "损坏"): m.rollback(backup)

    def test_bad_archive_paths_and_links_cannot_escape(self):
        for idx, name in enumerate(("../outside", "/tmp/p2p-outside", "safe/../../outside")):
            arc = self.root / f"bad{idx}.tar.gz"
            with tarfile.open(arc, "w:gz") as t:
                data = b"bad"; info = tarfile.TarInfo(name); info.size = len(data); t.addfile(info, io.BytesIO(data))
            with self.assertRaises(m.InstallError): m.safe_extract(arc, self.root / f"extract{idx}")
        arc = self.root / "link.tar.gz"
        with tarfile.open(arc, "w:gz") as t:
            info = tarfile.TarInfo("link"); info.type = tarfile.SYMTYPE; info.linkname = "/tmp"; t.addfile(info)
        with self.assertRaises(m.InstallError): m.safe_extract(arc, self.root / "extract-link")

    def test_corrupt_payload_rejected(self):
        payload = self.root / "source.tar.gz"; payload.write_bytes(b"original")
        m.write_json(self.root / "manifest.json", {"schema": 1, "version": "0.1.45",
            "payload_sha256": {payload.name: m.sha(payload)}})
        payload.write_bytes(b"damaged")
        with self.assertRaisesRegex(m.InstallError, "SHA-256"): m.verify_payload(self.root)

    def test_skipped_gpu_test_is_failure(self):
        with self.assertRaisesRegex(m.InstallError, "跳过"):
            m.shell_run([sys.executable, "-c", "raise SystemExit(77)"])

    def test_cuda_uuid_mapping_and_bidirectional_peer_requirement(self):
        first = 'GPU-11111111-1111-1111-1111-111111111111'
        second = 'GPU-22222222-2222-2222-2222-222222222222'
        csv = f'0, NVIDIA GeForce RTX 2080 Ti, {first}, 7.5, 22528, 1000\n1, NVIDIA GeForce RTX 2080 Ti, {second}, 7.5, 22528, 1000\n'
        class Driver:
            allowed = True
            calls = []
            def cuInit(self, flags): return 0
            def cuDeviceGetCount(self, out): out._obj.value = 2; return 0
            def cuDeviceGet(self, out, ordinal): out._obj.value = ordinal + 10; return 0
            def cuDeviceGetUuid(self, out, device):
                # Reverse the driver's device order relative to nvidia-smi.
                value = '22' if device == 10 else '11'
                ctypes.memmove(out, bytes.fromhex(value * 16), 16); return 0
            def cuDeviceCanAccessPeer(self, out, a, b):
                self.calls.append((a,b)); out._obj.value = int(self.allowed); return 0
        driver = Driver()
        with patch.object(m, 'shell_run', side_effect=[csv, 'GPU0 X NV2\nGPU1 NV2 X\n']), patch.object(m.ctypes, 'CDLL', return_value=driver):
            result = m.gpu_preflight([0,1])
        self.assertTrue(result['nvlink_detected'])
        self.assertEqual(driver.calls, [(11,10),(10,11)])
        driver.allowed = False
        with patch.object(m, 'shell_run', return_value=csv), patch.object(m.ctypes, 'CDLL', return_value=driver):
            with self.assertRaisesRegex(m.InstallError, '双向 P2P'):
                m.gpu_preflight([0,1])

    def test_build_or_gpu_failure_never_reaches_config_commit(self):
        for failing_step in ("prerequisite_tools", "gpu_preflight", "build_engine"):
            with self.subTest(step=failing_step), contextlib.ExitStack() as stack:
                for name, value in (("verify_payload", {"bundle_id": "test"}),
                                    ("prerequisite_tools", "/fake/nvcc"),
                                    ("gpu_preflight", {"interconnect": "NV2", "nvlink_detected": True}),
                                    ("source_tree", self.root / "source"), ("build_engine", self.root / "build"),
                                    ("install_files", None)):
                    stack.enter_context(patch.object(m, name, side_effect=m.InstallError("fixture failure")
                                                     if name == failing_step else None, return_value=value))
                commit = stack.enter_context(patch.object(m, "commit_config"))
                stack.enter_context(patch.object(m.os, "nice"))
                buf = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                with self.assertRaises(m.InstallError):
                    m.main(["--strata-dir", str(self.root), "--config", str(self.config)])
                commit.assert_not_called()
                self.assertNotIn("private-fixture-key", buf.getvalue())
                self.assertEqual(self.config.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
