"""Shared micro-averaged VisualProbe split diagnostics."""
SPLITS = ('visualprobe_easy', 'visualprobe_medium', 'visualprobe_hard')


def subset_summaries(records, pending=False):
    result = {}
    for name in SPLITS:
        rows = [r for r in records if r.get('subset') == name or
                r.get('data_source') == name.replace('visualprobe_', 'visual_probe_')]
        if not rows:
            continue
        n = len(rows)
        correct = sum(bool(r.get('correct')) for r in rows)
        valid = sum(bool(r.get('format_valid')) for r in rows)
        result[name] = dict(n=n, correct=None if pending else correct,
            accuracy=None if pending else correct/n,
            correct_deterministic=sum(bool(r.get('deterministic_correct')) for r in rows),
            accuracy_deterministic=sum(bool(r.get('deterministic_correct')) for r in rows)/n,
            format_valid=valid, format_missing=n-valid, format_rate=valid/n,
            judge_status='pending' if pending else 'scored')
    if sum(r['n'] for r in result.values()) != len(records):
        raise ValueError('VisualProbe records have missing or ambiguous subset identities')
    return result
