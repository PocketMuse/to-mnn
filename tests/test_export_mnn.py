import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.sd15 import export_mnn


class MnnExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.onnx = self.root / "onnx"
        self.reference = self.root / "reference"
        self.output = self.root / "w8"
        self.reference.mkdir()
        for name in ("text_encoder", "unet", "vae_decoder"):
            (self.onnx / name).mkdir(parents=True)
            (self.onnx / name / "model.onnx").write_bytes(b"onnx")
            (self.reference / f"{name}.mnn").write_bytes(name.encode())
        self.log = "Use HQQ to quant weight\nConverted Success!\n"
        self.returncode = 0
        self.payload = b"converted"
        self.commands = []
        runner = patch.object(export_mnn.subprocess, "run", self.run_conversion)
        runner.start()
        self.addCleanup(runner.stop)

    def run_conversion(self, command, *, stdout, stderr, env, check):
        self.commands.append(command)
        self.assertEqual(env["PIP_NO_INDEX"], "1")
        self.assertEqual(env["PIP_DISABLE_PIP_VERSION_CHECK"], "1")
        stdout.write(self.log)
        output = Path(command[command.index("--MNNModel") + 1])
        if self.payload is not None:
            output.write_bytes(self.payload)
        return subprocess.CompletedProcess(command, self.returncode)

    def run_cli(self, *options):
        with patch(
            "sys.argv",
            [
                "export_mnn",
                "--onnx-dir",
                str(self.onnx),
                "--output-dir",
                str(self.output),
                *options,
            ],
        ):
            export_mnn.main()

    def run_w8(self):
        self.run_cli(
            "--unet-quantization",
            "hqq-b128",
            "--reference-mnn-dir",
            str(self.reference),
        )

    def test_w8_copies_float_models_and_quantizes_only_unet(self):
        sidecar = self.reference / "text_encoder.mnn.weight"
        sidecar.write_bytes(b"external weights")
        self.run_w8()
        self.assertEqual(len(self.commands), 1)
        command = self.commands[0]
        self.assertNotIn("--fp16", command)
        self.assertIn("--weightQuantAsymmetric=1", command)
        self.assertIn("--hqq", command)
        self.assertEqual(command[command.index("--weightQuantBits") + 1], "8")
        self.assertEqual(command[command.index("--weightQuantBlock") + 1], "128")
        for name in ("text_encoder", "vae_decoder"):
            self.assertEqual((self.output / f"{name}.mnn").read_bytes(), name.encode())
        self.assertEqual(
            (self.output / sidecar.name).read_bytes(), sidecar.read_bytes()
        )
        self.assertEqual((self.output / "unet.mnn").read_bytes(), self.payload)
        self.assertEqual((self.output / "unet.convert.log").read_text(), self.log)

    def test_float_export_still_converts_all_components(self):
        self.run_cli()
        self.assertEqual(len(self.commands), 3)
        self.assertTrue(all("--fp16" in command for command in self.commands))

    def test_w8_requires_hqq_log_marker(self):
        self.log = "Converted Success!\n"
        with self.assertRaisesRegex(RuntimeError, "HQQ"):
            self.run_w8()

    def test_w8_rejects_converter_failure(self):
        self.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "conversion failed"):
            self.run_w8()

    def test_w8_requires_success_log_marker(self):
        self.log = "Use HQQ to quant weight\n"
        with self.assertRaisesRegex(RuntimeError, "conversion failed"):
            self.run_w8()

    def test_w8_rejects_missing_or_empty_output(self):
        for payload in (None, b""):
            with self.subTest(payload=payload):
                self.payload = payload
                with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                    self.run_w8()

    def test_w8_checks_reference_inputs_before_conversion(self):
        (self.reference / "vae_decoder.mnn").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_w8()
        self.assertEqual(self.commands, [])

    def test_w8_does_not_overwrite_reference_directory(self):
        self.output = self.reference
        with self.assertRaisesRegex(ValueError, "reference"):
            self.run_w8()
        self.assertEqual((self.reference / "unet.mnn").read_bytes(), b"unet")
        self.assertEqual(self.commands, [])


if __name__ == "__main__":
    unittest.main()
