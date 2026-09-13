import ast
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class Stage1ContractTests(unittest.TestCase):
    def test_system_b_uses_system_a_report_upload_field(self):
        system_a = (ROOT / "system_a" / "core" / "app_fastapi.py").read_text(encoding="utf-8")
        system_b = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")

        self.assertIn("async def report_endpoint(file: UploadFile = File(...))", system_a)
        self.assertIn('files={"file":', system_b)

    def test_system_b_source_remains_parseable(self):
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        ast.parse(source)

    def test_config_rejects_non_finite_and_out_of_range_values(self):
        sys.path.insert(0, str(ROOT / "system_b" / "core"))
        from config import ConfigManager

        manager = ConfigManager.__new__(ConfigManager)
        valid, errors = manager.validate_params({"GREEN_ADV": float("nan")})
        self.assertFalse(valid)
        self.assertTrue(any("有限数值" in error for error in errors))

        valid, errors = manager.validate_params({"GREEN_ADV": 999})
        self.assertFalse(valid)
        self.assertTrue(any("超出范围" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
