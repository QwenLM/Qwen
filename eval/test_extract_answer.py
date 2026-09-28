import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_eval():
    for name in ("torch", "jsonlines", "datasets", "transformers", "transformers.generation"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["datasets"].load_from_disk = lambda *args, **kwargs: None
    sys.modules["datasets"].load_dataset = lambda *args, **kwargs: None
    sys.modules["transformers"].AutoModelForCausalLM = object
    sys.modules["transformers"].AutoTokenizer = object
    sys.modules["transformers.generation"].GenerationConfig = object
    path = Path(__file__).resolve().parent / "evaluate_gsm8k.py"
    spec = importlib.util.spec_from_file_location("evaluate_gsm8k", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExtractAnswerTest(unittest.TestCase):
    def test_a_thousands_comma_stays_in_the_number(self):
        extract = load_eval().extract_answer
        self.assertEqual(extract("#### 1,000"), 1000)
        self.assertEqual(extract("the total is 1,234"), 1234)
        self.assertEqual(extract("42"), 42)


if __name__ == "__main__":
    unittest.main()
