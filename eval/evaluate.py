"""Evaluate one merged full RL checkpoint with replicated vLLM engines.

Unlike the P4 sweep worker, this entry point never materializes visual-only
weights.  ``--model-path`` must already be a complete Hugging Face checkpoint,
as produced by EasyR1's FSDP model merger.

``--output`` names a run directory containing ``summary.jsonl``, per-question
records under ``samples/``, and replicated vLLM logs under ``logs/``.
"""
from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path
import sys
_PROJECT_DIR = Path(__file__).resolve().parent
if Path(__file__).resolve().parent != _PROJECT_DIR:
    sys.path.insert(0, str(_PROJECT_DIR))
from eval_paths import DATA_ROOT, apply_data_config, add_data_arguments, validate_data_paths
import p4_eval_vllm as vclient
import vllm_runtime as multi
import extra_perception_benchmarks as extra_bench
import no_tag_llm_judge
DEFAULT_BENCHMARKS = 'zoombench,vstar_all,hrbench4k,mmerealworld'
EXTRA_TOKEN_BENCHMARKS = frozenset({'babyvision', 'perceptionbench', 'o3bench'})

def benchmark_args(args, benchmark):
    """Independent request budget, also used for resume/cache signatures."""
    result = argparse.Namespace(**vars(args))
    override = getattr(args, 'extra_max_new_tokens', None)
    if benchmark in EXTRA_TOKEN_BENCHMARKS and override is not None:
        result.max_new_tokens = override
    return result

def parse_bool(value: str | bool) -> bool:
    """Parse explicit shell booleans such as ``--use_cot true``."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'true', '1', 'yes', 'y'}:
        return True
    if normalized in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'expected true or false, got {value!r}')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--run-name', required=True, help='Stable label written as the output arm, e.g. rl_base_10k_e2.')
    parser.add_argument('--step', type=int, default=None, help='Optional bookkeeping step. By default it is inferred from the nearest global_step_<N> component in --model-path. Defaults to 0 when no step component is present.')
    parser.add_argument('--gpus', default='')
    parser.add_argument('--tp', type=int, default=1, help='GPUs per replica; GPU count must be divisible by TP')
    parser.add_argument('--vllm-bin', default=None)
    parser.add_argument('--max-num-seqs', type=int, default=16)
    parser.add_argument('--enable-chunked-prefill', action='store_true', help='Explicitly enable chunked prefill for non-GLM engines')
    parser.add_argument('--extra-max-new-tokens', type=int, default=None, help='Override output budget for babyvision, perceptionbench, o3bench only')
    parser.add_argument('--defer-judge', action='store_true', help='Save VisualProbe and extra perception outputs with pending/null scores; no LLM judge calls')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--benchmarks', default=DEFAULT_BENCHMARKS, help='Comma list including babyvision, perceptionbench, o3bench, cvbench, mmstar, mmvp, and visualprobe_easy, visualprobe_medium, visualprobe_hard, or visualprobe_all (pooled three splits).')
    parser.add_argument('--output', required=True, help='Run output directory. It will contain summary.jsonl, samples/, and logs/. For an old command, remove only the trailing .jsonl.')
    parser.add_argument('--per-engine-concurrency', type=int, default=12)
    parser.add_argument('--answer-format', choices=vclient.ANSWER_FORMATS, default='answer_tag')
    parser.add_argument('--generation-profile', choices=vclient.GENERATION_PROFILES, default=vclient.QWEN3VL_GENERATION_PROFILE, help='qwen3vl preserves existing Qwen3-VL evaluation. qwen35_nothinking changes only the Qwen3.5 chat-template switch while benchmark prompt construction stays on the Qwen3-VL path. qwen35_thinking enables Qwen3.5 native thinking, no ZwZ system prompt, and a user-level final <answer> instruction.')
    parser.add_argument('--max-new-tokens', type=int, default=None, help='Default: 8192 for either Qwen3.5 profile; otherwise 512 for boxed/answer-tag CoT and 8 for direct.')
    parser.add_argument('--min-tokens', type=int, default=None, help='Forbid EOS before this many generated tokens (vLLM min_tokens). Diagnostic/mitigation for empty-output RL checkpoints; apply uniformly across arms when comparing.')
    parser.add_argument('--mm-text-placeholder', default=None, help="Write this literal placeholder into the user text instead of letting vLLM prepend it and join with a newline. Pass 'qwen' as shorthand for '<|vision_start|><|image_pad|><|vision_end|>'. Default (unset) keeps the historical prompt, which carries an extra newline after <|vision_end|> that training never sees.")
    parser.add_argument('--visualprobe-root', default=str(DATA_ROOT / 'visualprobe'), help='Root containing the three downloaded VisualProbe splits.')
    parser.add_argument('--cvbench-root', default=str(DATA_ROOT / 'cvbench'), help='Root containing test_2d.parquet and test_3d.parquet.')
    parser.add_argument('--mmstar-root', default=str(DATA_ROOT / 'mmstar'), help='Root containing the downloaded mmstar.parquet.')
    parser.add_argument('--mmvp-root', default=str(DATA_ROOT / 'mmvp'), help="Root containing Questions.csv and the 'MMVP Images' directory.")
    parser.add_argument('--visualprobe-max-new-tokens', type=int, default=vclient.VISUALPROBE_DEFAULT_MAX_NEW_TOKENS, help='Minimum direct-generation budget for VisualProbe; boxed/answer-tag CoT continues to use --max-new-tokens.')
    parser.add_argument('--use_cot', '--use-cot', type=parse_bool, default=True, help="Model-agnostic CoT profile. true removes benchmark-specific direct/only directives. With answer_tag, ZwZ's original prompt is sent as a real system message (default: true).")
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max-model-len', type=int, default=32768)
    parser.add_argument('--max-num-batched-tokens', type=int, default=None, help='Prefill token budget per engine step; defaults to --max-model-len so multimodal prompts are never chunk-split (see vllm_runtime).')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--startup-timeout', type=int, default=900)
    parser.add_argument('--preview-samples', type=int, default=3, help='Print this many representative generated samples after each benchmark (default: 3; use 0 to disable).')
    extra_bench.add_arguments(parser)
    vclient.add_visualprobe_judge_arguments(parser)
    no_tag_llm_judge.add_arguments(parser)
    add_data_arguments(parser)
    return parser

def require_merged_hf_model(model_path: str | Path) -> Path:
    model = Path(model_path).resolve()
    if not model.is_dir():
        raise FileNotFoundError(f'RL model directory does not exist: {model}')
    if not (model / 'config.json').is_file():
        raise FileNotFoundError(f'RL model is missing config.json: {model}')
    has_weights = (model / 'model.safetensors').is_file() or (model / 'model.safetensors.index.json').is_file() or any(model.glob('model-*.safetensors'))
    if not has_weights:
        actor = model.parent if model.name == 'huggingface' else model
        raise FileNotFoundError(f"{model} has config/tokenizer files but no merged model weights. Run EasyR1/scripts/model_merger.py with --local_dir {actor}, then pass {actor / 'huggingface'}.")
    return model
GLOBAL_STEP_COMPONENT = re.compile('^global_step_(\\d+)$')

def infer_step_from_model_path(model_path: str | Path) -> int | None:
    """Read the nearest ``global_step_<N>`` component from a model path."""
    original = Path(model_path)
    candidates = [original]
    resolved = original.resolve()
    if resolved != original:
        candidates.append(resolved)
    for candidate in candidates:
        for component in reversed(candidate.parts):
            match = GLOBAL_STEP_COMPONENT.fullmatch(component)
            if match:
                return int(match.group(1))
    return None

def resolve_step(model_path: str | Path, explicit_step: int | None) -> int:
    """Infer the bookkeeping step, or validate an explicit override."""
    if explicit_step is not None and explicit_step < 0:
        raise ValueError('--step must be non-negative')
    inferred_step = infer_step_from_model_path(model_path)
    if explicit_step is None:
        return inferred_step if inferred_step is not None else 0
    if inferred_step is not None and explicit_step != inferred_step:
        raise ValueError(f'--step {explicit_step} conflicts with the global_step_{inferred_step} component in --model-path.')
    return explicit_step

def completed_benchmarks(output: Path, *, run_name: str, step: int, answer_format: str, use_cot: bool=False, generation_profile: str=vclient.QWEN3VL_GENERATION_PROFILE) -> set[str]:
    completed = set()
    if not output.exists():
        return completed
    with output.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('arm') == run_name and row.get('step') == step and (row.get('answer_format', 'direct') == answer_format) and (bool(row.get('use_cot', False)) == use_cot) and (row.get('generation_profile', vclient.QWEN3VL_GENERATION_PROFILE) == generation_profile):
                completed.add(str(row['benchmark']))
    return completed

def select_preview_records(records: list[dict], count: int, *, required_qtypes: tuple[str, ...]=()) -> list[dict]:
    """Choose deterministic, informative examples for console inspection."""
    if count <= 0:
        return []
    selected: list[dict] = []
    selected_ids: set[int] = set()

    def add_first(predicate) -> None:
        for record in records:
            record_id = id(record)
            if record_id not in selected_ids and predicate(record):
                selected.append(record)
                selected_ids.add(record_id)
                return
    add_first(lambda record: record.get('format_valid', record.get('answer_tag_format_valid', record.get('boxed_format_valid'))) is False)
    if len(selected) < count:
        add_first(lambda record: bool(record.get('correct')))
    if len(selected) < count:
        add_first(lambda record: not bool(record.get('correct')))
    for record in records:
        if len(selected) >= count:
            break
        if id(record) not in selected_ids:
            selected.append(record)
            selected_ids.add(id(record))
    for qtype in required_qtypes:
        if any((record.get('qtype') == qtype for record in selected)):
            continue
        candidate = next((record for record in records if record.get('qtype') == qtype), None)
        if candidate is None:
            continue
        if len(selected) < count:
            selected.append(candidate)
            selected_ids.add(id(candidate))
        elif selected:
            selected_ids.discard(id(selected[-1]))
            selected[-1] = candidate
            selected_ids.add(id(candidate))
    return selected[:count]

def preview_payload(benchmark: str, record: dict) -> dict:
    """Keep the console preview compact while retaining scoring evidence."""
    identifier = next((record[key] for key in ('question_id', 'doc_id', 'idx', 'index') if key in record), None)
    payload = {'benchmark': benchmark, 'id': identifier}
    for key in ('cycle', 'qtype', 'category', 'l2_category', 'type', 'task', 'source', 'source_dataset', 'question', 'choices', 'pred_choice', 'gt_choice', 'match_rule', 'strict_correct', 'prompt', 'generated'):
        if key in record:
            payload[key] = record[key]
    payload.update({'boxed_answer': record.get('boxed_answer'), 'pred': record.get('pred'), 'gt': record.get('gt'), 'correct': bool(record.get('correct')), 'boxed_format_valid': record.get('boxed_format_valid')})
    for key in ('answer_tag_answer', 'answer_tag_format_valid', 'format_valid'):
        if key in record:
            payload[key] = record[key]
    return payload

def print_sample_previews(benchmark: str, records: list[dict], count: int) -> None:
    previews = select_preview_records(records, count, required_qtypes=('blank',) if benchmark == 'zoombench' else ())
    if not previews:
        return
    block_separator = '=' * 100
    sample_separator = '-' * 100
    print(f'\n{block_separator}\nSAMPLE PREVIEW | benchmark={benchmark} | count={len(previews)}\n{block_separator}', flush=True)
    for (number, record) in enumerate(previews, start=1):
        if number > 1:
            print(sample_separator, flush=True)
        print(f'[sample {number}] ' + json.dumps(preview_payload(benchmark, record), ensure_ascii=False), flush=True)
    print(f'{block_separator}\n', flush=True)

def main() -> None:
    args = build_parser().parse_args()
    apply_data_config(args)
    if args.dry_run:
        args.defer_judge = True
    if args.per_engine_concurrency <= 0:
        raise ValueError('--per-engine-concurrency must be positive')
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError('--gpu-memory-utilization must be in (0, 1]')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')
    if args.max_new_tokens is None:
        args.max_new_tokens = vclient.default_max_new_tokens(args)
    vclient.MIN_TOKENS = args.min_tokens
    if args.mm_text_placeholder == 'qwen':
        args.mm_text_placeholder = vclient.QWEN_VISION_TEXT_PLACEHOLDER
    vclient.MM_TEXT_PLACEHOLDER = args.mm_text_placeholder
    if args.max_new_tokens <= 0:
        raise ValueError('--max-new-tokens must be positive')
    if args.extra_max_new_tokens is not None and args.extra_max_new_tokens <= 0:
        raise ValueError('--extra-max-new-tokens must be positive')
    if args.visualprobe_max_new_tokens <= 0:
        raise ValueError('--visualprobe-max-new-tokens must be positive')
    if args.use_cot and args.answer_format == 'direct':
        raise ValueError('--use_cot true requires a non-direct answer format')
    if args.answer_format == 'answer_tag' and (not args.use_cot):
        raise ValueError('--answer-format answer_tag requires --use_cot true')
    vclient.validate_generation_options(args)
    no_tag_llm_judge.preflight(args)
    if args.preview_samples < 0:
        raise ValueError('--preview-samples must be non-negative')
    selected = [x.strip() for x in args.benchmarks.split(',') if x.strip()]
    if {'cvbench', 'mmstar'}.intersection(selected):
        try:
            import pyarrow.parquet
        except ImportError as exc:
            raise RuntimeError("CVBench/MMStar require pyarrow in the evaluator's Python environment; install it before launching GPUs") from exc
    extra_bench.preflight(args, selected)
    if args.extra_bench_preflight_only:
        if not any((x in extra_bench.NAMES for x in selected)):
            raise ValueError('Select babyvision, perceptionbench or o3bench for preflight')
        return
    model = require_merged_hf_model(args.model_path)
    args.step = resolve_step(args.model_path, args.step)
    gpus = multi.parse_gpus(args.gpus)
    if args.tp < 1 or len(gpus) % args.tp:
        raise ValueError('GPU count must be divisible by positive --tp')
    benchmarks = [item.strip() for item in args.benchmarks.split(',') if item.strip()]
    if not benchmarks:
        raise ValueError('--benchmarks selected no benchmarks')
    if 'visualprobe_all' in benchmarks and any((name in benchmarks for name in vclient.VISUALPROBE_SPLITS)):
        raise ValueError('Use visualprobe_all OR individual VisualProbe splits, not both')
    if args.dry_run:
        validate_data_paths(args, benchmarks)
        print(json.dumps({'status': 'ok', 'mode': 'dry-run', 'model': str(model), 'benchmarks': benchmarks, 'gpus': gpus}))
        return
    requested_output = Path(args.output)
    layout = multi.prepare_output_layout(requested_output, args.use_cot)
    args.logs_dir = str(layout.logs_dir)
    args.output = str(layout.summary)
    if layout.run_dir != requested_output:
        print(f'use_cot=true: output directory renamed to {layout.run_dir}', flush=True)
    summary_display = str(layout.summary.resolve())
    print(f'SUMMARY_JSONL: {summary_display}', flush=True)
    done = completed_benchmarks(layout.summary, run_name=args.run_name, step=args.step, answer_format=args.answer_format, use_cot=args.use_cot, generation_profile=vclient.generation_profile_for(args))
    if not args.defer_judge and layout.summary.exists():
        for line in layout.summary.read_text().splitlines():
            old = json.loads(line)
            if old.get('judge_status') == 'pending':
                done.discard(old.get('benchmark'))
    if layout.summary.exists():
        for line in layout.summary.read_text().splitlines():
            old = json.loads(line)
            benchmark = old.get('benchmark')
            if benchmark in done and benchmark in benchmarks and (args.generation_profile == 'glm53_default') and (args.answer_format == 'answer_tag') and (old.get('glm53_prompt_revision') != 'zwz_tag_reminder_v1'):
                raise RuntimeError('Old GLM prompt results exist; use a new --output directory')
            if benchmark in done and benchmark in benchmarks and (old.get('arm') == args.run_name) and (old.get('step') == args.step) and (old.get('answer_format') == args.answer_format) and (old.get('use_cot') == args.use_cot) and (old.get('generation_profile') == args.generation_profile) and (old.get('max_new_tokens') != benchmark_args(args, benchmark).max_new_tokens):
                raise RuntimeError(f'{benchmark}: completed output has a different token budget; use a new --output directory')
    for benchmark in extra_bench.NAMES.intersection(done, benchmarks):
        signature = extra_bench.completion_signature(benchmark_args(args, benchmark), benchmark)
        matching = False
        if layout.summary.exists():
            for line in layout.summary.read_text().splitlines():
                row = json.loads(line)
                if row.get('benchmark') == benchmark and row.get('arm') == args.run_name and (row.get('step') == args.step) and (row.get('answer_format') == args.answer_format) and (row.get('use_cot') == args.use_cot) and (row.get('generation_profile') == args.generation_profile) and (row.get('evaluation_signature') == signature) and (not extra_bench.cache_has_aborted_generations(row.get('sample_cache'))):
                    matching = True
        if not matching:
            done.discard(benchmark)
    pending = [benchmark for benchmark in benchmarks if benchmark not in done]
    if not pending:
        print('all requested RL benchmarks are already complete', flush=True)
        if args.judge_when_no_ans_tag:
            no_tag_llm_judge.print_table(layout.summary, args.run_name, args.step, benchmarks)
        print(f'SUMMARY_JSONL: {summary_display}', flush=True)
        return
    safe_name = re.sub('[^A-Za-z0-9_.-]+', '_', args.run_name).strip('_')
    if not safe_name:
        raise ValueError('--run-name must contain a usable character')
    cot_tag = 'cot' if args.use_cot else 'nocot'
    generation_profile = vclient.generation_profile_for(args)
    generation_tag = {vclient.QWEN3VL_GENERATION_PROFILE: '', vclient.QWEN35_NOTHINKING_GENERATION_PROFILE: '-qwen35-nothinking', vclient.QWEN35_GENERATION_PROFILE: '-qwen35-thinking', 'glm53_default': '-glm53-default'}[generation_profile]
    served_name = f'{safe_name}-{args.step}-{args.answer_format}-{cot_tag}{generation_tag}'
    print(f"RL evaluator: model={model} run={args.run_name}:{args.step} profile={args.answer_format} generation_profile={vclient.generation_profile_for(args)} max_new_tokens={args.max_new_tokens} use_cot={str(args.use_cot).lower()} gpus={','.join(gpus)}", flush=True)
    multi.install_teardown_signal_handlers()
    replicas: list[multi.Replica] = []
    try:
        replicas = multi.start_replicas(str(model), served_name, gpus, args.port, args)
        multi.wait_replicas_ready(replicas, args.startup_timeout)
        client_args = multi.client_args_for(replicas, served_name, args.run_name, args.step, args)
        for key in ('model_path', 'min_tokens', 'mm_text_placeholder', 'babyvision_root', 'perceptionbench_file', 'o3bench_root', 'extra_judge_config', 'extra_judge_api_keys', 'extra_judge_model', 'extra_judge_endpoint', 'extra_judge_concurrency', 'extra_judge_max_tokens', 'extra_benchmark_metadata', 'defer_judge'):
            if hasattr(args, key):
                setattr(client_args, key, getattr(args, key))
        for benchmark in pending:
            started = time.perf_counter()
            client_args.max_new_tokens = benchmark_args(args, benchmark).max_new_tokens
            print(f'{benchmark}: max_new_tokens={client_args.max_new_tokens}', flush=True)
            if benchmark in extra_bench.NAMES:
                (summary, records) = extra_bench.evaluate(benchmark, client_args, vclient)
            else:
                if args.defer_judge and benchmark.startswith('visualprobe_'):
                    client_args.vp_judge = False
                (summary, records) = multi.evaluate_benchmark(benchmark, client_args)
                if args.defer_judge and benchmark.startswith('visualprobe_'):
                    for record in records:
                        record.update(correct=None, judge_status='pending')
                    summary.update(accuracy=None, correct=None, judge_status='pending', metric_profile='llm_judge_pending')
                    if benchmark == 'visualprobe_all':
                        from visualprobe_groups import subset_summaries
                        summary['by_subset'] = subset_summaries(records, pending=True)
                        summary['by_type'] = {key: None for key in summary.get('by_type', {})}
            elapsed = time.perf_counter() - started
            if args.judge_when_no_ans_tag:
                no_tag_llm_judge.apply(benchmark, summary, records, args, vclient)
            row = {'arm': args.run_name, 'step': args.step, 'benchmark': benchmark, 'engine': 'vllm-multigpu-replicated-full-rl', 'existing_endpoint': None, 'glm53_prompt_revision': 'zwz_tag_reminder_v1' if args.generation_profile == 'glm53_default' and args.answer_format == 'answer_tag' else None, 'answer_tag_reminder': vclient.GLM53_ANSWER_TAG_REMINDER if args.generation_profile == 'glm53_default' and args.answer_format == 'answer_tag' else None, 'api_model': None, 'api_image_policy': None, 'thinking_policy': None, 'model_path': str(model), 'num_engines': len(replicas), 'tensor_parallel_size': args.tp, 'defer_judge': args.defer_judge, 'gpus': gpus, 'per_engine_concurrency': args.per_engine_concurrency, 'answer_format': args.answer_format, 'use_cot': args.use_cot, 'generation_profile': vclient.generation_profile_for(args), 'prompt_profile': vclient.prompt_profile_for(args), 'system_prompt': vclient.system_prompt_for(args), 'boxed_instruction': vclient.BOXED_ANSWER_INSTRUCTION if args.answer_format == 'boxed' else None, 'max_new_tokens': client_args.max_new_tokens, 'extra_max_new_tokens': args.extra_max_new_tokens, 'min_tokens': args.min_tokens, 'mm_text_placeholder': args.mm_text_placeholder, 'temperature': 1.0 if 'local' == 'api' and args.model_path == 'gpt-5.5' else 0.0, 'elapsed_sec': round(elapsed, 1), **summary}
            error_count = 0
            for record in records:
                failure = record.get('evaluation_refusal') or {}
                if failure.get('code') == 'generation_api_error':
                    record['evaluation_error'] = failure
                    record['error'] = True
                    error_count += 1
            row['error_count'] = error_count
            row['error_question_n'] = len(records)
            row['error_rate'] = error_count / len(records) if records else 0.0
            dump = layout.samples_dir / (f'{safe_name}_{args.step}_{benchmark}_{args.answer_format}' + f'_{cot_tag}{generation_tag}' + '_samples.jsonl')
            dump_tmp = dump.with_suffix(dump.suffix + '.tmp')
            with dump_tmp.open('w', encoding='utf-8') as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            dump_tmp.replace(dump)
            multi.append_jsonl_atomically(layout.summary, row)
            print(f"[{args.run_name}:{args.step}] {benchmark}: {(format(summary['accuracy'], '.2%') if summary['accuracy'] is not None else 'judge pending')} (n={summary['n']}) format={summary.get('format_rate', summary.get('boxed_format_rate', 1.0)):.2%} errors={error_count}/{len(records)} in {elapsed / 60:.1f}min -> {dump.name}", flush=True)
            if args.judge_when_no_ans_tag:
                print(no_tag_llm_judge.report_line(args.run_name, args.step, benchmark, summary), flush=True)
            print_sample_previews(benchmark, records, args.preview_samples)
    finally:
        multi.teardown_replicas(replicas)
        if replicas and 'local' != 'existing':
            print(f'released {len(replicas)} RL engines', flush=True)
    if args.judge_when_no_ans_tag:
        no_tag_llm_judge.print_table(layout.summary, args.run_name, args.step, benchmarks)
    print('RL EVALUATOR DONE', flush=True)
    print(f'SUMMARY_JSONL: {summary_display}', flush=True)
if __name__ == '__main__':
    main()
