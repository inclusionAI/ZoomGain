"""Local vLLM replica lifecycle and benchmark dispatch for merged RL checkpoints."""
from __future__ import annotations
from eval_paths import DATA_ROOT
import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import p4_eval_vllm as vclient
import p4_vllm_worker as single_worker
ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Replica:
    gpu: str
    port: int
    endpoint: str
    process: subprocess.Popen
    log_handle: object
    log_path: Path

@dataclass(frozen=True)
class OutputLayout:
    """All artifacts belonging to one evaluator invocation."""
    run_dir: Path
    summary: Path
    samples_dir: Path
    logs_dir: Path

def prepare_output_layout(output: str | Path, cot_enabled: bool) -> OutputLayout:
    """Resolve and create the run-scoped output tree.

    ``--output`` intentionally names a directory now. Rejecting the historical
    ``.jsonl`` spelling avoids silently creating a directory whose name looks
    like a file and gives existing commands a precise one-character-class fix:
    remove the suffix.
    """
    requested = Path(output)
    if requested.suffix.lower() == '.jsonl':
        raise ValueError(f"--output now names a directory, not a JSONL file; remove '.jsonl': --output {requested.with_suffix('')}")
    run_dir = vclient.output_path_for_profile(requested, cot_enabled)
    if run_dir.exists() and (not run_dir.is_dir()):
        raise NotADirectoryError(f'--output must be a directory, but an existing file was found: {run_dir}')
    layout = OutputLayout(run_dir=run_dir, summary=run_dir / 'summary.jsonl', samples_dir=run_dir / 'samples', logs_dir=run_dir / 'logs')
    layout.samples_dir.mkdir(parents=True, exist_ok=True)
    layout.logs_dir.mkdir(parents=True, exist_ok=True)
    if layout.summary.exists():
        if not layout.summary.is_file():
            raise IsADirectoryError(f'expected summary JSONL file, found: {layout.summary}')
    else:
        layout.summary.touch()
    return layout

def parse_gpus(spec: str) -> list[str]:
    """Parse ``0236`` or ``0,2,3,6`` into unique physical GPU identifiers."""
    compact = str(spec).strip()
    if not compact:
        raise ValueError('--gpus must not be empty')
    if re.search('[,\\s]', compact):
        gpus = [part for part in re.split('[,\\s]+', compact) if part]
    elif compact.isdigit():
        gpus = list(compact)
    else:
        raise ValueError(f"invalid --gpus {spec!r}; use compact '0236' or '0,2,3,6'")
    if any((not gpu.isdigit() for gpu in gpus)):
        raise ValueError(f'invalid GPU identifier in --gpus {spec!r}')
    if len(set(gpus)) != len(gpus):
        raise ValueError(f'duplicate GPU identifier in --gpus {spec!r}')
    return gpus

def ensure_ports_free(ports: list[int], stray_pids: list[int] | None=None) -> None:
    sockets = []
    try:
        for port in ports:
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            handle.bind(('127.0.0.1', port))
            sockets.append(handle)
    except OSError as exc:
        hint = ''
        if stray_pids:
            hint = f"\nOrphaned engines from an earlier run are present; they are the likely holder. Free them with:\n  kill -TERM {' '.join((str(pid) for pid in stray_pids))}"
        raise RuntimeError(f'multi-GPU vLLM port is unavailable: {exc}{hint}') from exc
    finally:
        for handle in sockets:
            handle.close()

def open_unique_log(path: Path) -> tuple[Path, object]:
    """Create a new log without overwriting an earlier/resumed invocation."""
    attempt = 1
    while True:
        candidate = path if attempt == 1 else path.with_name(f'{path.stem}_attempt{attempt}{path.suffix}')
        try:
            return (candidate, candidate.open('x'))
        except FileExistsError:
            attempt += 1

def vllm_serve_command(model_dir: str, served_name: str, port: int, args) -> list[str]:
    """Build one replica command while preserving the Qwen3-VL argv exactly."""
    command = [getattr(args, 'vllm_bin', None) or single_worker.VLLM_PY, 'serve', model_dir, '--served-model-name', served_name, '--port', str(port), '--max-model-len', str(args.max_model_len), '--gpu-memory-utilization', str(args.gpu_memory_utilization), '--trust-remote-code', '--gdn-prefill-backend', getattr(args, 'gdn_prefill_backend', 'triton'), '--max-num-batched-tokens', str(getattr(args, 'max_num_batched_tokens', None) or args.max_model_len), '--no-enable-chunked-prefill']
    tp = getattr(args, 'tp', 1)
    if getattr(args, 'enable_chunked_prefill', False):
        command[command.index('--no-enable-chunked-prefill')] = '--enable-chunked-prefill'
    if tp != 1:
        command.extend(['--tensor-parallel-size', str(tp)])
    if vclient.generation_profile_for(args) == 'glm53_default':
        index = command.index('--gdn-prefill-backend')
        del command[index:index + 2]
        if '--no-enable-chunked-prefill' in command:
            command[command.index('--no-enable-chunked-prefill')] = '--enable-chunked-prefill'
        command.extend(['--max-num-seqs', str(getattr(args, 'max_num_seqs', 16)), '--no-enable-prefix-caching', '--no-enable-flashinfer-autotune', '--reasoning-parser', 'glm45', '--kv-cache-dtype', 'bfloat16', '--attention-config', '{"sparse_mla_force_mqa":true}'])
    if vclient.generation_profile_for(args) in vclient.QWEN35_GENERATION_PROFILES:
        command.extend(['--reasoning-parser', 'qwen3'])
    if vclient.generation_profile_for(args) != 'glm53_default' and getattr(args, 'max_num_seqs', None):
        command.extend(['--max-num-seqs', str(args.max_num_seqs)])
    if getattr(args, 'extra_max_images', None):
        command.extend(['--limit-mm-per-prompt', json.dumps({'image': args.extra_max_images})])
    return command

def append_jsonl_atomically(path: Path, row: dict) -> None:
    """Append one row via replacement, including for legacy read-only files."""
    temporary = path.with_name(f'.{path.name}.tmp')
    with temporary.open('w', encoding='utf-8') as output_handle:
        if path.exists():
            with path.open(encoding='utf-8') as input_handle:
                shutil.copyfileobj(input_handle, output_handle)
        output_handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)

def start_replicas(model_dir: str, served_name: str, gpus: list[str], first_port: int, args) -> list[Replica]:
    tp = getattr(args, 'tp', 1)
    if tp < 1 or len(gpus) % tp:
        raise ValueError('GPU count must be divisible by positive TP')
    if vclient.generation_profile_for(args) == 'glm53_default':
        usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        for line in usage.splitlines():
            (index, used) = [x.strip() for x in line.split(',')]
            if index in gpus and int(used) > 100:
                raise RuntimeError(f'GPU {index} is occupied ({used} MiB); nothing was stopped')
    gpus = [','.join(gpus[i:i + tp]) for i in range(0, len(gpus), tp)]
    ports = [first_port + offset for offset in range(len(gpus))]
    ensure_ports_free(ports, getattr(args, '_orphaned_engine_pids', None))
    replicas = []
    logs_dir = Path(getattr(args, 'logs_dir', Path(args.output).parent))
    logs_dir.mkdir(parents=True, exist_ok=True)
    try:
        for (gpu, port) in zip(gpus, ports):
            requested_log_path = logs_dir / f'serve_{served_name}_gpu{gpu}_{port}.log'
            (log_path, log_handle) = open_unique_log(requested_log_path)
            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}
            executable = getattr(args, 'vllm_bin', None) or single_worker.VLLM_PY
            environment['PATH'] = f"{Path(executable).parent}:{environment.get('PATH', '')}"
            if vclient.generation_profile_for(args) != 'glm53_default':
                environment.setdefault('VLLM_USE_DEEP_GEMM', '0')
            try:
                process = subprocess.Popen(vllm_serve_command(model_dir, served_name, port, args), stdout=log_handle, stderr=subprocess.STDOUT, env=environment, preexec_fn=os.setsid)
            except BaseException:
                log_handle.close()
                raise
            replica = Replica(gpu=gpu, port=port, endpoint=f'http://127.0.0.1:{port}/v1', process=process, log_handle=log_handle, log_path=log_path)
            replicas.append(replica)
            _LIVE_REPLICAS.append(replica)
            print(f'[{served_name}] launch gpu={gpu} port={port} log={log_path.name}', flush=True)
    except BaseException:
        teardown_replicas(replicas)
        raise
    return replicas

def wait_replicas_ready(replicas: list[Replica], timeout: int) -> None:
    for replica in replicas:
        try:
            single_worker.wait_ready(replica.port, timeout, replica.process)
        except Exception as exc:
            raise RuntimeError(f'vLLM replica gpu={replica.gpu} port={replica.port} failed; inspect {replica.log_path}') from exc
        print(f'  ready gpu={replica.gpu} endpoint={replica.endpoint}', flush=True)
_LIVE_REPLICAS: list[Replica] = []

def teardown_replicas(replicas: list[Replica]) -> None:
    for replica in replicas:
        if replica.process.poll() is None:
            try:
                os.killpg(os.getpgid(replica.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if all((replica.process.poll() is not None for replica in replicas)):
            break
        time.sleep(0.2)
    for replica in replicas:
        if replica.process.poll() is None:
            try:
                os.killpg(os.getpgid(replica.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                replica.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        replica.log_handle.close()
        try:
            _LIVE_REPLICAS.remove(replica)
        except ValueError:
            pass

def client_args_for(replicas: list[Replica], served_name: str, arm: str, step: int, args):
    return argparse.Namespace(endpoints=[replica.endpoint for replica in replicas], endpoint=replicas[0].endpoint, per_engine_concurrency=args.per_engine_concurrency, model=served_name, arm=arm, step=step, zoombench_annotation=getattr(args, 'zoombench_annotation', str(DATA_ROOT / 'zoombench/annotation_zoombench.json')), zoombench_image_root=getattr(args, 'zoombench_image_root', str(DATA_ROOT / 'zoombench')), vstar_annotation=getattr(args, 'vstar_annotation', str(DATA_ROOT / 'vstar/test_questions.jsonl')), vstar_image_root=getattr(args, 'vstar_image_root', str(DATA_ROOT / 'vstar/images')), hrbench_root=getattr(args, 'hrbench_root', str(DATA_ROOT / 'hrbench')), mmerealworld_annotation=getattr(args, 'mmerealworld_annotation', str(DATA_ROOT / 'MME-RW/data/MME-RealWorld-Lite.json')), mmerealworld_image_root=getattr(args, 'mmerealworld_image_root', str(DATA_ROOT / 'MME-RW/data/imgs')), treebench_annotation=getattr(args, 'treebench_annotation', str(DATA_ROOT / 'treebench/TreeBench.tsv')), babyvision_root=str(getattr(args, 'babyvision_root', DATA_ROOT / 'babyvision/babyvision_data')), cvbench_root=str(getattr(args, 'cvbench_root', DATA_ROOT / 'cvbench')), mmstar_root=str(getattr(args, 'mmstar_root', DATA_ROOT / 'mmstar')), mmvp_root=str(getattr(args, 'mmvp_root', DATA_ROOT / 'mmvp')), visualprobe_root=str(getattr(args, 'visualprobe_root', DATA_ROOT / 'visualprobe')), limit=args.limit, max_new_tokens=args.max_new_tokens, visualprobe_max_new_tokens=int(getattr(args, 'visualprobe_max_new_tokens', vclient.VISUALPROBE_DEFAULT_MAX_NEW_TOKENS)), answer_format=args.answer_format, use_cot=args.use_cot, generation_profile=vclient.generation_profile_for(args), concurrency=args.per_engine_concurrency, mme_concurrency=args.per_engine_concurrency, output=args.output, **vclient.visualprobe_judge_args(args))

def evaluate_benchmark(benchmark: str, client_args):
    if benchmark == 'zoombench':
        return vclient.eval_zoombench(client_args)
    if benchmark in vclient.VSTAR_BENCHMARKS:
        return vclient.eval_vstar(client_args, benchmark)
    if benchmark in ('hrbench4k', 'hrbench8k'):
        return vclient.eval_hrbench(client_args, benchmark[-2:])
    if benchmark == 'mmerealworld':
        return vclient.eval_mmerealworld(client_args)
    if benchmark == 'treebench':
        return vclient.eval_treebench(client_args)
    if benchmark == 'babyvision':
        return vclient.eval_babyvision(client_args)
    if benchmark == 'cvbench':
        return vclient.eval_cvbench(client_args)
    if benchmark == 'mmstar':
        return vclient.eval_mmstar(client_args)
    if benchmark == 'mmvp':
        return vclient.eval_mmvp(client_args)
    if benchmark in vclient.VISUALPROBE_BENCHMARKS:
        return vclient.eval_visualprobe(client_args, benchmark)
    raise ValueError(f'unknown benchmark {benchmark!r}')

def _handle_terminating_signal(signum, _frame) -> None:
    print(
        f"\nreceived {signal.Signals(signum).name}: releasing "
        f"{len(_LIVE_REPLICAS)} vLLM engine(s) before exit",
        flush=True,
    )
    teardown_replicas(list(_LIVE_REPLICAS))
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)

def install_teardown_signal_handlers() -> None:
    """Release engines on ``kill``, a closed tmux window, or a dropped SSH.

    SIGINT already unwinds through the ``finally`` in :func:`main`; SIGTERM and
    SIGHUP kill the interpreter outright, which is exactly how the 2026-07
    evaluations leaked engines for 12 and 17 days.
    """
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _handle_terminating_signal)
