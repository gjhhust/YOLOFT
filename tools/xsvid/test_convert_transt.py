import unittest
import torch

from tools.xsvid.convert_transt import portable_state


class ConversionTests(unittest.TestCase):
    def fixture(self):
        return {"net": {("class_embed." if index < 170 else "backbone.") + str(index):
                        torch.tensor([float(index)]) for index in range(430)}}

    def test_tensor_clone(self):
        source = self.fixture()
        result = portable_state(source)
        self.assertEqual(len(result), 430)
        self.assertTrue(torch.equal(result["class_embed.0"], source["net"]["class_embed.0"]))
        self.assertNotEqual(result["class_embed.0"].data_ptr(), source["net"]["class_embed.0"].data_ptr())

    def test_missing_head(self):
        source = self.fixture()
        source["net"]["backbone.extra"] = source["net"].pop("class_embed.0")
        with self.assertRaises(ValueError):
            portable_state(source)

    def test_nonfinite(self):
        source = self.fixture()
        source["net"]["class_embed.0"].fill_(float("nan"))
        with self.assertRaises(ValueError):
            portable_state(source)

    def test_non_tensor(self):
        source = self.fixture()
        source["net"]["class_embed.0"] = "metadata"
        with self.assertRaises(ValueError):
            portable_state(source)


if __name__ == "__main__":
    unittest.main()
