from pathlib import Path
import unittest

from src.transaction_ml_pipeline import run_pipeline


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_creates_required_artifacts(self) -> None:
        result = run_pipeline()
        project_root = Path(__file__).resolve().parents[1]

        self.assertGreater(result["best_cluster_count"], 1)
        self.assertTrue((project_root / "models" / "model_clustering.joblib").exists())
        self.assertTrue((project_root / "models" / "decision_tree_model.h5").exists())
        self.assertTrue((project_root / "data" / "processed" / "transactions_training_with_target.csv").exists())


if __name__ == "__main__":
    unittest.main()
