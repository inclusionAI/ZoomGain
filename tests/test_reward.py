"""Regression checks for the public training reward. Run: python3 -m unittest discover -s tests."""
import json
import unittest
from zoomgain.reward import compute_score
from zoomgain.scoring import thresholded_anls

def score(text, row):
    return compute_score("zwz_perception", text, row["answer"],
        {"comparison_dataset":"zoomgain", "zoomgain_scoring_row_json":json.dumps(row)})["score"]

class RewardTests(unittest.TestCase):
    def test_mcq_parsing(self):
        row={"question_type":"attribute", "answer_format":"multiple_choice",
             "question":"Color?\nA. Red\nB. Green", "answer":"B"}
        self.assertEqual(score("<answer>B</answer>",row),1)
        self.assertEqual(score("<answer>A</answer>",row),0)
        self.assertEqual(score("B",row),0)
        self.assertEqual(score("<answer>A or B</answer>",row),0)
        self.assertEqual(score("<answer>A</answer><answer>B</answer>",row),1)
    def test_counting(self):
        row={"question_type":"counting","answer_format":"open_ended","answer":"2"}
        self.assertEqual(score("<answer>2.0</answer>",row),1)
        self.assertEqual(score("<answer>3</answer>",row),0)
    def test_ocr_training_policy(self):
        row={"question_type":"ocr","answer_format":"open_ended","answer":"ABC DEF",
             "scoring":{"similarity_threshold":0.7,"case_sensitive":True}}
        self.assertEqual(score("<answer>abcdef</answer>",row),1)
        self.assertEqual(score("<answer>XYZ</answer>",row),0)
        self.assertAlmostEqual(thresholded_anls("1234567xxx","1234567890")[0],0.7)
    def test_metadata_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Ground truth"):
            compute_score("zwz_perception","<answer>B</answer>","A",
                {"comparison_dataset":"zoomgain","zoomgain_scoring_row_json":json.dumps({"answer":"B"})})

if __name__ == "__main__":
    unittest.main()
