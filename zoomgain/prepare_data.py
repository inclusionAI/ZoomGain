"""Validate prepared ZoomGain parquet; optionally remap images for a downloaded dataset.

Relative image paths are resolved against the input Parquet's directory.
Use --output to write a separate Parquet with absolute image paths.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--old-image-root", type=Path)
    args = parser.parse_args()
    if args.old_image_root and not args.image_root:
        parser.error("--old-image-root requires --image-root")
    if args.image_root and not args.output:
        parser.error("image remapping requires --output")
    if args.output and args.output.resolve() == args.input.resolve():
        parser.error("output must differ from input")
    table = pq.read_table(args.input)
    rows = table.to_pylist()
    if not rows:
        raise ValueError("empty training dataset")
    input_dir = args.input.resolve().parent
    checked = set()
    categories = {}
    for i, row in enumerate(rows):
        for key in ("prompt", "images", "reward_model", "extra_info", "data_source"):
            if key not in row:
                raise ValueError(f"row {i}: missing {key}")
        info = row["extra_info"]
        if info.get("comparison_dataset") != "zoomgain":
            raise ValueError(f"row {i}: expected comparison_dataset=zoomgain")
        scoring = json.loads(info["zoomgain_scoring_row_json"])
        if str(scoring["answer"]) != str(row["reward_model"]["ground_truth"]):
            raise ValueError(f"row {i}: inconsistent reference answer")
        category = scoring["question_type"]
        if category not in ("attribute", "attribute_recognition", "counting", "ocr") and scoring["answer_format"] != "multiple_choice":
            raise ValueError(f"row {i}: unsupported category {category}")
        categories[category] = categories.get(category, 0) + 1
        if sum(m["content"].count("<image>") for m in row["prompt"]) != len(row["images"]):
            raise ValueError(f"row {i}: image placeholders do not match image count")
        paths = []
        for image in row["images"]:
            path = Path(image)
            if args.image_root:
                if args.old_image_root:
                    path = args.image_root / path.relative_to(args.old_image_root)
                elif path.is_absolute():
                    raise ValueError("absolute image paths require --old-image-root for remapping")
                else:
                    path = args.image_root / path
            elif not path.is_absolute():
                path = input_dir / path
            path = path.resolve()
            if path not in checked:
                if not path.is_file():
                    raise FileNotFoundError(f"row {i}: missing image {path}")
                checked.add(path)
            paths.append(str(path))
        row["images"] = paths
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), args.output)
    print(json.dumps({"rows":len(rows), "unique_images":len(checked), "categories":categories, "status":"ok"}))

if __name__ == "__main__":
    main()
