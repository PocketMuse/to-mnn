"""선택적 Windows C++ tokenizer와 HuggingFace CLIP의 입력 ID 비교."""

import os
import subprocess
import unittest


@unittest.skipUnless(
    os.environ.get("SD15_TOKENIZER_TEST") and os.environ.get("SD15_TOKENIZER_DIR"),
    "Native tokenizer and assets not selected",
)
class NativeTokenizerTest(unittest.TestCase):
    def test_clip_token_ids(self):
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(
            os.environ["SD15_TOKENIZER_DIR"], local_files_only=True
        )
        for prompt in [
            "",
            "a photo of a cat sitting on a wooden table, natural light",
            " A PHOTO\t of\n a CAT ",
            "It's a cat! We're here, aren't we?",
            "12345 + 67.89",
            "blurry, low quality, deformed",
            "cat " * 100,
            "<|startoftext|>a cat<|endoftext|>",
            "한글 고양이 猫",
            "Café, déjà vu! 🐈",
        ]:
            with self.subTest(prompt=prompt[:80]):
                result = subprocess.run(
                    [
                        os.environ["SD15_TOKENIZER_TEST"],
                        "--tokenizer-dir",
                        os.environ["SD15_TOKENIZER_DIR"],
                        "--stdin",
                    ],
                    input=prompt,
                    encoding="utf-8",
                    check=True,
                    capture_output=True,
                )
                actual = [int(value) for value in result.stdout.split()]
                expected = tokenizer(
                    prompt, padding="max_length", truncation=True, max_length=77
                ).input_ids
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
