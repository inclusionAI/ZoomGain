"""CPU-only regression checks for the extracted evaluation runtime."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"eval"))
sys.path.insert(0,str(ROOT/"eval_ZoomGainBench"))
import vllm_runtime as multi
import evaluate as rl
import benchmark_core as bench

class EvaluationPackagingTests(unittest.TestCase):
    def test_replica_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(tp=1, logs_dir=tmp, output=tmp, generation_profile="qwen35_nothinking", vllm_bin="vllm")
            proc=MagicMock()
            proc.poll.return_value=0
            with patch.object(multi,"ensure_ports_free"), patch.object(multi,"vllm_serve_command",return_value=["vllm"]), patch.object(multi.subprocess,"Popen",return_value=proc):
                replicas=multi.start_replicas("model","test",["0"],18840,args)
                self.assertEqual(len(replicas),1)
                self.assertIn(replicas[0],multi._LIVE_REPLICAS)
                multi.teardown_replicas(replicas)
                self.assertNotIn(replicas[0],multi._LIVE_REPLICAS)
                self.assertTrue(replicas[0].log_handle.closed)
    def test_layout_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout=multi.prepare_output_layout(Path(tmp)/"run",True)
            self.assertTrue(layout.samples_dir.is_dir())
            args=rl.build_parser().parse_args(["--model-path","model","--run-name","test","--port","18840","--output",tmp,"--zoombench-annotation","custom.json"])
            client=multi.client_args_for([SimpleNamespace(endpoint="http://localhost:18840/v1")],"model","test",0,args)
            self.assertEqual(client.zoombench_annotation,"custom.json")
    def test_benchmark_scoring(self):
        row={"answer_format":"multiple_choice","question_type":"attribute","question":"Color?\nA. Red\nB. Blue","answer":"B"}
        self.assertEqual(bench.score_answer("B",row)["score"],1)
        self.assertEqual(bench.score_answer(None,row)["score"],0)
        self.assertAlmostEqual(bench.thresholded_anls("1234567xxx","1234567890")[0],0.7)

    def test_benchmark_packaged_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            Image.new("RGB", (4, 4)).save(root / "images" / "test.png")
            annotations = root / "annotations.jsonl"
            row = {
                "id": "example", "image_path": "images/test.png",
                "question": "Color?\nA. Red\nB. Blue",
                "question_zh": "什么颜色？\nA. Red\nB. Blue",
                "answer": "B", "answer_text": "Blue", "evidence_bbox": [0, 0, 4, 4],
                "answer_format": "multiple_choice", "question_type": "attribute",
                "scoring": {"metric": "exact_match"},
            }
            annotations.write_text(json.dumps(row) + "\n", encoding="utf-8")
            tasks = bench.expand_language_tasks(bench.load_annotations(annotations), "all")
            self.assertEqual([task["question"] for task in tasks], [row["question_zh"], row["question"]])
            self.assertEqual(Path(tasks[0]["image_path"]), root.resolve() / "images" / "test.png")
            for field in ("question_zh", "evidence_bbox"):
                incomplete = {key: value for key, value in row.items() if key != field}
                annotations.write_text(json.dumps(incomplete) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, field):
                    bench.load_annotations(annotations)
            row["evidence_bbox"] = [0, 0, 5, 4]
            annotations.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "evidence_bbox"):
                bench.load_annotations(annotations)

if __name__=="__main__":unittest.main()
