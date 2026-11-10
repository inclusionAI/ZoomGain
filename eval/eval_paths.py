"""Portable dataset locations for the local model evaluation suite."""
import json
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("EVAL_DATA_ROOT", Path(__file__).resolve().parent / "data")).expanduser().resolve()
DEFAULT_PATHS = {'zoombench_annotation': 'zoombench/annotation_zoombench.json', 'zoombench_image_root': 'zoombench', 'vstar_annotation': 'vstar/test_questions.jsonl', 'vstar_image_root': 'vstar/images', 'hrbench_root': 'hrbench', 'mmerealworld_annotation': 'MME-RW/data/MME-RealWorld-Lite.json', 'mmerealworld_image_root': 'MME-RW/data/imgs', 'treebench_annotation': 'treebench/TreeBench.tsv', 'cvbench_root': 'cvbench', 'mmstar_root': 'mmstar', 'mmvp_root': 'mmvp', 'visualprobe_root': 'visualprobe', 'babyvision_root': 'BabyVision/babyvision_data', 'perceptionbench_file': 'PerceptionBench/PerceptionBench.jsonl', 'o3bench_root': 'O3-Bench/test'}

def add_data_arguments(parser):
    parser.add_argument("--data-config", type=Path, help="JSON mapping dataset argument names to paths; relative paths are resolved beside the JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate model and dataset locations without GPU inference or judge calls.")
    for name, rel in DEFAULT_PATHS.items():
        flag = "--" + name.replace("_", "-")
        if flag not in parser._option_string_actions:
            parser.add_argument(flag, default=str(DATA_ROOT / rel))

def apply_data_config(args):
    if not args.data_config:
        return
    config = args.data_config.expanduser().resolve()
    values = json.loads(config.read_text())
    unknown = set(values) - set(DEFAULT_PATHS)
    if unknown:
        raise ValueError(f"Unknown dataset paths: {sorted(unknown)}")
    for name, value in values.items():
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = config.parent / p
        setattr(args, name, str(p.resolve()))

def validate_data_paths(args, benchmarks):
    mapping = {
        "zoombench":["zoombench_annotation","zoombench_image_root"],
        "vstar":["vstar_annotation","vstar_image_root"],
        "vstar_all":["vstar_annotation","vstar_image_root"],
        "vstar_relpos":["vstar_annotation","vstar_image_root"],
        "hrbench4k":["hrbench_root"],"hrbench8k":["hrbench_root"],
        "mmerealworld":["mmerealworld_annotation","mmerealworld_image_root"],
        "treebench":["treebench_annotation"], "cvbench":["cvbench_root"],
        "mmstar":["mmstar_root"], "mmvp":["mmvp_root"],
        "visualprobe_all":["visualprobe_root"], "visualprobe_easy":["visualprobe_root"],
        "visualprobe_medium":["visualprobe_root"], "visualprobe_hard":["visualprobe_root"],
        "babyvision":["babyvision_root"], "perceptionbench":["perceptionbench_file"],
        "o3bench":["o3bench_root"],
    }
    for benchmark in benchmarks:
        if benchmark not in mapping:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        for name in mapping[benchmark]:
            p=Path(getattr(args,name))
            if not p.exists():
                raise FileNotFoundError(f"{benchmark}: {name} does not exist: {p}")
