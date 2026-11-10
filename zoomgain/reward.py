"""ZoomGain RL reward, preserving the released 9K training policy."""
import json
import os
from zoomgain import scoring

def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  format_reward_weight=0.0, **kwargs):
    info = extra_info or {}
    dataset = info.get('comparison_dataset')
    if dataset != 'zoomgain':
        raise ValueError('Missing/invalid comparison_dataset metadata; use the supplied train parquet')
    weight = float(format_reward_weight)
    if not 0 <= weight <= 1:
        raise ValueError('format_reward_weight must be in [0,1]')
    row = json.loads(info['zoomgain_scoring_row_json'])
    if str(row['answer']) != str(ground_truth):
        raise ValueError('Ground truth and ZoomGain scoring metadata disagree')
    # RL policy: OCR ignores case regardless of older parquet metadata.
    # Keep the native ANLS threshold and whitespace handling unchanged.
    if row["question_type"] == "ocr" and row["answer_format"] != "multiple_choice":
        row["scoring"] = dict(row["scoring"], case_sensitive=False)
    answer, _ = scoring.extract_last_answer_tag(solution_str)
    scored = scoring.score_answer(answer, row)
    accuracy = float(scored['score'])
    valid = float(answer is not None)
    # Preserve the native ZoomGain evaluator: a complete final tag remains
    # scoreable even on a length-limited response.
    limit = int(os.environ.get('ZOOMGAIN_MAX_RESPONSE_TOKENS', '2048'))
    truncated = float(info.get('response_length', 0) >= limit)
    return {'score': (1 - weight) * accuracy + weight * valid, 'accuracy': accuracy,
            'format': valid, 'accuracy_reward': (1 - weight) * accuracy,
            'format_reward': weight * valid, 'truncated': truncated}
