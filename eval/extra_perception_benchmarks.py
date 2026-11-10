"""BabyVision, PerceptionBench and O3-Bench with an external GPT judge.

Uses this project's inference profile, single greedy pass, original image
resolution (BabyVision retains its existing EXIF/RGB normalization). Scores
are internal comparisons, not reproductions of the official leaderboards.
"""
from __future__ import annotations

from eval_paths import DATA_ROOT
import base64
import hashlib
import json
import os
import re
import random
import queue
from email.utils import parsedate_to_datetime
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm.auto import tqdm

NAMES = {'babyvision', 'perceptionbench', 'o3bench'}
ROOT = DATA_ROOT
SCORING_VERSION = 'gpt51_extra_bench_v2'


def add_arguments(parser):
    parser.add_argument('--babyvision-root', default=str(ROOT/'BabyVision/babyvision_data'))
    parser.add_argument('--perceptionbench-file', default=str(ROOT/'PerceptionBench/PerceptionBench.jsonl'))
    parser.add_argument('--o3bench-root', default=str(ROOT/'O3-Bench/test'))
    parser.add_argument('--extra-judge-config', default=None,
                        help='Private JSON: base_url, model, api_key. Environment EXTRA_JUDGE_* overrides it.')
    parser.add_argument('--extra-judge-api-keys', default='api_keys.txt')
    parser.add_argument('--extra-judge-model', default='Qwen3.8-Flash-Next')
    parser.add_argument('--extra-judge-endpoint', default='http://127.0.0.1:8031/v1')
    parser.add_argument('--extra-judge-concurrency', type=int, default=32)
    parser.add_argument('--extra-judge-max-tokens', type=int, default=2048)
    parser.add_argument('--extra-bench-preflight-only', action='store_true',
                        help='Validate selected extra datasets and judge config without GPU/API calls.')


def judge_config(args):
    if getattr(args, 'defer_judge', False):
        return {'model': 'pending', 'base_url': ''}
    if not getattr(args, 'extra_judge_config', None) and not os.environ.get('EXTRA_JUDGE_API_KEY'):
        path = Path(args.extra_judge_api_keys)
        keys = [s.strip() for s in path.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError('Judge key pool must be nonempty and contain no duplicate keys')
        endpoint = args.extra_judge_endpoint.rstrip('/').removesuffix('/chat/completions')
        if not endpoint.startswith(('https://', 'http://')):
            raise ValueError('Judge endpoint must use HTTP or HTTPS')
        return dict(model=args.extra_judge_model, base_url=endpoint, label='key_pool', _keys=keys)
    path = Path(args.extra_judge_config) if getattr(args, 'extra_judge_config', None) else ROOT/'.judge_gpt51.json'
    config = json.loads(path.read_text()) if path.is_file() else {}
    for field in ('api_key', 'base_url', 'model'):
        config[field] = os.environ.get('EXTRA_JUDGE_'+field.upper(), config.get(field))
    config['base_url'] = (config.get('base_url') or 'http://127.0.0.1:8031/v1').rstrip('/')
    config['model'] = config.get('model') or 'gpt-5.1'
    if not config.get('api_key'):
        raise ValueError('Set EXTRA_JUDGE_API_KEY or provide --extra-judge-config')
    if not config['base_url'].startswith(('https://', 'http://')):
        raise ValueError('The external judge endpoint must use HTTPS')
    fallback = config.get('fallback')
    if fallback:
        if not fallback.get('api_key') or not fallback.get('model'):
            raise ValueError('Fallback judge requires api_key and model')
        fallback.setdefault('base_url', config['base_url'])
        if not fallback['base_url'].startswith('https://'):
            raise ValueError('Fallback judge endpoint must use HTTPS')
    return config


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_data(name, args):
    import babyvision_data as baby
    if name == 'babyvision':
        ann = Path(args.babyvision_root)/'meta_data.jsonl'
        data = [dict(id=str(s['index']), question=baby.build_babyvision_question(s),
                     gt=s['answer'], category=s['type'], subtype=s['subtype'],
                     options=s['options'], images=[s['image']], hint='', baby=True)
                for s in baby.load_babyvision(args.babyvision_root)]
    else:
        ann = Path(args.perceptionbench_file) if name == 'perceptionbench' else Path(args.o3bench_root)/'metadata.jsonl'
        data = []
        with ann.open() as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                s = json.loads(line)
                if name == 'perceptionbench':
                    gt = s['answer']
                    gt = gt['answer'] if isinstance(gt, dict) else gt
                    images = s.get('image', [])
                    if not isinstance(images, list) or not images or any(not isinstance(x,str) or not x.startswith('data:image/') for x in images):
                        raise ValueError(f'Invalid embedded images in PerceptionBench row {i}')
                    data.append(dict(id=str(s['index']), question=s['problem'], gt=str(gt),
                                     category=s['error_category'], images=images,
                                     hint=s.get('system') or s.get('hint') or '', options=''))
                else:
                    img = Path(args.o3bench_root)/s['file_name']
                    if not img.is_file():
                        raise FileNotFoundError(img)
                    if not s.get('options') or not re.fullmatch('[A-Z]', str(s['answer'])):
                        raise ValueError(f'Unsupported O3 answer schema row {i}')
                    data.append(dict(id=str(i), question=s['question'], options=s['options'],
                                     gt=s['answer'], category=s['subset'], images=[str(img)], hint=''))
    if not data or len({s['id'] for s in data}) != len(data):
        raise ValueError(f'Empty or duplicate IDs in {name}')
    if args.limit is not None:
        data = data[:args.limit]
    return data, {'annotation':str(ann), 'annotation_sha256':file_hash(ann),
                  'n':len(data), 'max_images':max(len(s['images']) for s in data)}


def preflight(args, names):
    selected = [n for n in names if n in NAMES]
    if not selected:
        return {}
    if args.extra_judge_concurrency < 1 or args.extra_judge_max_tokens < 1:
        raise ValueError('Judge concurrency and token budget must be positive')
    config = judge_config(args)
    metadata = {}
    for name in selected:
        data, info = load_data(name, args)
        metadata[name] = info
        del data
    args.extra_benchmark_metadata = metadata
    args.extra_max_images = max(x['max_images'] for x in metadata.values())
    print('Extra benchmark preflight: '+json.dumps(metadata), flush=True)
    print('External judge: '+config['model']+' '+config['base_url'], flush=True)
    if '_keys' in config:
        print(f"Judge key pool: {len(config['_keys'])} keys; active slots={min(args.extra_judge_concurrency,len(config['_keys']))}; one judgement per key", flush=True)
    return metadata


def retry_after_seconds(value):
    try:
        return max(0.0, float(value))
    except (ValueError, TypeError):
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return 0.0


def post_json(endpoint, payload, api_key=None, timeout=180):
    headers = {'Content-Type':'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer '+api_key
    error = 'unknown'
    attempts = 12 if api_key else 4
    for attempt in range(attempts):
        server_wait = 0.0
        try:
            req = urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',
                                         data=json.dumps(payload).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            error = f'HTTP {exc.code}'  # Never log credentials or response bodies.
            if exc.code not in (408,409,429,500,502,503,504):
                break
            server_wait = retry_after_seconds(exc.headers.get('Retry-After')) if exc.headers else 0.0
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            error = type(exc).__name__
        if api_key:
            delay = max(server_wait, min(15 * 2**attempt, 120)) + random.uniform(0, 5)
            if attempt < attempts-1:
                time.sleep(delay)
        elif attempt < attempts-1:
            time.sleep(min(2**attempt, 8))
    raise RuntimeError('API request failed: '+error)


def build_messages(sample, name, args, vclient):
    import babyvision_data as baby
    images = []
    for value in sample['images']:
        if value.startswith('data:image/'):
            images.append(value)
        else:
            encoded, media = (baby.encode_babyvision_image(value) if name=='babyvision'
                              else vclient.encode_original_image(Path(value)))
            images.append(f'data:{media};base64,{encoded}')
    question = sample['question']
    if name == 'o3bench':
        question += '\n'+sample['options']
    prompt = vclient.format_evaluation_prompt(question, args)
    content = []
    used = set()
    cursor = 0
    for match in re.finditer(r'<\|image_(\d+)\|>', prompt):
        index = int(match.group(1))-1
        if not 0 <= index < len(images):
            raise ValueError('Image placeholder out of bounds')
        if match.start()>cursor:
            content.append({'type':'text','text':prompt[cursor:match.start()]})
        content.append({'type':'image_url','image_url':{'url':images[index]}})
        used.add(index)
        cursor=match.end()
    if cursor < len(prompt):
        content.append({'type':'text','text':prompt[cursor:]})
    remaining=[{'type':'image_url','image_url':{'url':img}} for i,img in enumerate(images) if i not in used]
    content=remaining+content
    if vclient.MM_TEXT_PLACEHOLDER:
        # The explicit Qwen placeholder option is only defined for single-image input.
        if len(images)!=1:
            raise ValueError('--mm-text-placeholder cannot be used with multi-image questions')
        for part in content:
            if part['type']=='text':
                part['text']=vclient.MM_TEXT_PLACEHOLDER+part['text']; break
    system='\n\n'.join(x for x in (vclient.system_prompt_for(args),sample['hint']) if x)
    messages=([{'role':'system','content':system}] if system else [])
    messages.append({'role':'user','content':content})
    return messages, prompt


def judge_prompt(name, record):
    templates=Path(__file__).with_name('extra_benchmark_prompts')
    if name=='babyvision':
        return (templates/'babyvision_judge.txt').read_text().format(
            question=record['question'], groundtruth=record['gt'], modeloutput=record['pred'])
    if name=='perceptionbench':
        text=(templates/'perception_judge.txt').read_text()
        # Replace only original template fields, never placeholders inside model output.
        values={'problem':record['question'],'reference_answer':record['gt'],'assistant_answer':record['pred']}
        return re.sub(r'\{(problem|reference_answer|assistant_answer)\}',lambda m:values[m[1]],text)
    return (templates/'o3_judge.txt').read_text().format(question=record['question'],options=record['options'],model_answer=record['pred'])


def parse_verdict(name, reply, record):
    if name=='o3bench':
        letter=reply.strip()
        valid=set(re.findall(r'(?:^|\s)([A-Z])[.)]',record['options']))
        if not valid: valid=set('ABCDEF')
        if letter not in valid:
            raise ValueError('Unparseable O3 judge choice')
        return letter==record['gt'], letter
    if name=='perceptionbench':
        match=re.search(r'\[judge\]\s*(true|false)\s*$',reply.strip(),re.I)
    else:
        match=re.fullmatch(r'\s*(true|false)\s*',reply,re.I)
    if not match:
        raise ValueError('Unparseable judge verdict')
    return match[1].lower()=='true', match[1].lower()


class JudgeAttemptFailure(RuntimeError):
    def __init__(self, reason, retry_after=0):
        super().__init__(reason)
        self.retry_after = retry_after


def judge_api_once(provider, prompt, max_tokens):
    payload = {'model': provider['model'],
               'messages': [{'role': 'user', 'content': prompt}],
               'max_completion_tokens': max_tokens}
    request = urllib.request.Request(provider['base_url'].rstrip('/')+'/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
        'Authorization': 'Bearer '+provider['api_key']})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        delay = retry_after_seconds(exc.headers.get('Retry-After')) if exc.headers else 0
        raise JudgeAttemptFailure(f'HTTP {exc.code}', delay) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise JudgeAttemptFailure(type(exc).__name__) from None


def judge_with_fallback(name, record, config, max_tokens):
    prompt = judge_prompt(name, record)
    # Each sample owns its retry loop. No shared cooldown or provider switching.
    for cycle in range(12):
        server_wait = 0.0
        for label, provider in [(config.get('label', 'X'), config),
                                (config['fallback'].get('label', 'Y'), config['fallback'])]:
            try:
                response = judge_api_once(provider, prompt, max_tokens)
                reply = response['choices'][0]['message'].get('content') or ''
                correct, verdict = parse_verdict(name, reply, record)
                return dict(record, correct=correct, judge_raw=reply,
                    judge_verdict=verdict, judge_status='ok', judge_provider=label,
                    judge_model=provider['model'], judge_endpoint=provider['base_url'])
            except (JudgeAttemptFailure, ValueError, KeyError, IndexError, TypeError) as exc:
                server_wait = max(server_wait, getattr(exc, 'retry_after', 0))
        if cycle < 11:
            delay = max(server_wait, min(15 * 2**cycle, 120)) + random.uniform(0, 5)
            time.sleep(delay)
    raise RuntimeError('Judge X and Y failed after 12 cycles; cached successes are preserved')


def judge_o3_extraction_fallback(record, config, max_tokens, history):
    # Extraction only: never include the ground truth in the judge request.
    prompt = ('Extract the options explicitly selected in the model answer. '
        'Do NOT solve the question or decide which option is true. '
        'Multiple selections are allowed in your extraction. '
        'If the answer explicitly selects A and C, return {"selected_options":["A","C"]}; '
        'do not force one letter. If no option matches, return {"selected_options":[]}. '
        'Output ONLY this JSON object, no explanation.\nOptions:\n' + record['options']
        + '\nModel answer:\n' + record['pred'])
    valid = set(re.findall(r'(?:^|\s)([A-Z])[.)]', record['options'])) or set('ABCDEF')
    budget = max_tokens
    for attempt in range(3):
        response = post_json(config['base_url'], dict(model=config['model'],
            messages=[dict(role='user', content=prompt)], max_completion_tokens=budget), config['api_key'])
        choice = response['choices'][0]
        reply = choice['message'].get('content') or ''
        history.append(dict(protocol='o3_selected_options_v1', max_tokens=budget,
            finish_reason=choice.get('finish_reason'), reply=reply, usage=response.get('usage')))
        if choice.get('finish_reason') == 'length':
            budget = min(max(max_tokens, 8192), max(4096, budget * 2))
            continue
        try:
            text = reply.strip()
            if text.startswith('```') and text.endswith('```'):
                text = re.sub(r'^```(?:json)?\s*', '', text)[:-3].strip()
            parsed = json.loads(text)
            selected = parsed['selected_options']
            if not isinstance(selected, list) or any(not isinstance(x, str) or x not in valid for x in selected):
                raise ValueError('Invalid extracted option list')
            selected = sorted(set(selected))
        except (ValueError, KeyError, TypeError):
            continue
        return dict(record, correct=(len(selected) == 1 and selected[0] == record['gt']),
            judge_raw=reply, judge_verdict=','.join(selected), judge_selected_options=selected,
            judge_status='ok', judge_protocol='o3_selected_options_v1',
            judge_provider=config.get('label', 'X'), judge_model=config['model'],
            judge_endpoint=config['base_url'], judge_effective_max_tokens=budget, judge_attempts=history)
    raise RuntimeError('O3 extraction fallback failed; not counted as wrong')


def judge_record(name, record, config, max_tokens):
    if record.get('evaluation_refusal'):
        return dict(record, correct=False, score=0.0, judge_raw='', judge_status='content_policy_violation')
    if name == 'o3bench' and record.get('finish_reason', 'stop') != 'stop':
        return dict(record, correct=False, judge_raw='', judge_status='incomplete_generation')
    if not record['pred'].strip():
        return dict(record, correct=False, judge_raw='', judge_status='empty_prediction')
    if config.get('fallback'):
        return judge_with_fallback(name, record, config, max_tokens)
    # Keep the prompt/parser unchanged; raise the budget only after a truncated
    # judge response. Successful low-budget judgements remain valid cache hits.
    budget = max_tokens
    ceiling = max(max_tokens, 8192)
    history = []
    for attempt in range(3):
        while True:
            response=post_json(config['base_url'], {'model':config['model'],
                'messages':[{'role':'user','content':judge_prompt(name,record)}],
                'max_completion_tokens':budget}, config['api_key'])
            choice=response['choices'][0]
            reply=choice['message'].get('content') or ''
            history.append(dict(max_tokens=budget, finish_reason=choice.get('finish_reason'),
                                usage=response.get('usage'), reply=reply))
            if choice.get('finish_reason') != 'length':
                break
            if name == 'o3bench':
                return judge_o3_extraction_fallback(record, config, max_tokens, history)
            if budget >= ceiling:
                raise RuntimeError(f'Judge output still truncated at {budget} tokens; not counted as wrong')
            budget = min(ceiling, max(4096, budget * 2))
        try:
            correct, verdict=parse_verdict(name,reply,record)
            return dict(record,correct=correct,judge_raw=reply,judge_verdict=verdict,judge_status='ok',
                        judge_provider=config.get('label', 'X'),judge_model=config['model'],judge_endpoint=config['base_url'],
                        judge_effective_max_tokens=budget,judge_attempts=history)
        except ValueError:
            if attempt==2:
                if name == 'o3bench':
                    return judge_o3_extraction_fallback(record, config, max_tokens, history)
                raise


def evaluate(name, args, vclient):
    data, metadata=load_data(name,args)
    config=judge_config(args)
    signature=dict(version=SCORING_VERSION, benchmark=name, data=metadata,
        model_path=args.model_path, model=args.model, step=args.step,
        answer_format=args.answer_format,use_cot=args.use_cot,
        generation_profile=args.generation_profile,max_new_tokens=args.max_new_tokens,
        min_tokens=args.min_tokens,placeholder=args.mm_text_placeholder,
        judge_model=config['model'],judge_endpoint=config['base_url'],judge_max_tokens=args.extra_judge_max_tokens)
    # Generation cache must survive switching from deferred to actual judging.
    legacy_digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:20]
    generation_signature={k:v for k,v in signature.items() if not k.startswith('judge_')}
    digest=hashlib.sha256(json.dumps(generation_signature,sort_keys=True).encode()).hexdigest()[:20]
    cache=Path(args.output).parent/'extra_benchmark_cache'/f'{name}_{digest}.jsonl'
    legacy_cache=cache.with_name(f'{name}_{legacy_digest}.jsonl')
    if not cache.exists() and legacy_cache.exists():
        cache=legacy_cache
    cache.parent.mkdir(exist_ok=True)
    saved={}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try:
                r=json.loads(line); saved[r['index']]=r
            except json.JSONDecodeError: continue # Interrupted last append is rerun.
    if cache.exists():
        repaired=cache.with_suffix('.repair.tmp')
        repaired.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in saved.values()))
        repaired.replace(cache)
    lock=threading.Lock()
    def persist(record):
        with lock, cache.open('a') as f:
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
    def generate(sample,endpoint):
        if sample['id'] in saved and saved[sample['id']].get('finish_reason') != 'abort':
            return saved[sample['id']]
        messages,prompt=build_messages(sample,name,args,vclient)
        payload={'model':args.model,'messages':messages,'max_tokens':args.max_new_tokens,'temperature':0.0}
        if args.min_tokens is not None: payload['min_tokens']=args.min_tokens
        if args.generation_profile in vclient.QWEN35_GENERATION_PROFILES:
            payload['chat_template_kwargs']={'enable_thinking':args.generation_profile==vclient.QWEN35_GENERATION_PROFILE}
        response=post_json(endpoint,payload,timeout=600)
        choice=response['choices'][0];message=choice['message']
        generated=str(message.get('reasoning') or message.get('reasoning_content') or '')+str(message.get('content') or '')
        pred,boxed=vclient.scoring_payload(generated,args)
        record=dict(index=sample['id'],question=sample['question'],options=sample['options'],
                    category=sample['category'],subtype=sample.get('subtype'),gt=sample['gt'],
                    pred=pred.strip(),generated=generated,prompt=prompt,finish_reason=choice.get('finish_reason'))
        record.update(content=message.get('content'),
            reasoning_content=message.get('reasoning_content') or message.get('reasoning'),
            usage=response.get('usage'), response_model=response.get('model'), response_id=response.get('id'),
            image_transport=response.get('_evaluation_image_transport'),
            evaluation_refusal=response.get('_evaluation_refusal'))
        vclient.add_boxed_record_fields(record,args=args,prompt=prompt,boxed_answer=boxed)
        persist(record)
        if record['finish_reason'] == 'abort':
            raise RuntimeError('Generation aborted; rerun to retry this sample from cache')
        return record
    records=vclient.parallel_eval_requests(generate,data,args,description=name,
                                           single_endpoint_concurrency=args.concurrency)
    if getattr(args, 'defer_judge', False):
        records = [dict(r, correct=False, score=0.0, judge_status='content_policy_violation')
                   if r.get('evaluation_refusal') else dict(r, correct=None, judge_status='pending')
                   for r in records]
        for record in records:
            persist(record)
        summary = dict(n=len(records), correct=None, accuracy=None,
            judge_status='pending', metric_profile='llm_judge_pending',
            dataset_metadata=metadata, sample_cache=str(cache),
            evaluation_signature=completion_signature(args,name))
        return vclient.add_boxed_summary(summary,records,args), records
    judge_signature = completion_signature(args,name)
    providers = queue.Queue()
    for key in config.get('_keys', []):
        providers.put(dict(model=config['model'],base_url=config['base_url'],label='key_pool',api_key=key))
    def reusable(record):
        return (record.get('judge_status') in ('empty_prediction','incomplete_generation','content_policy_violation')
                or (record.get('judge_status') == 'ok' and record.get('judge_signature') == judge_signature))
    def judge(record):
        provider = providers.get() if '_keys' in config else config
        try:
            clean = {k:v for k,v in record.items() if not k.startswith('judge_') and k not in ('correct','score')}
            result=judge_record(name,clean,provider,args.extra_judge_max_tokens)
        finally:
            if '_keys' in config:
                providers.put(provider)
        result['judge_signature'] = judge_signature
        if result.get('judge_status') == 'ok':
            result.setdefault('judge_model',provider['model'])
            result.setdefault('judge_endpoint',provider['base_url'])
            result.setdefault('judge_provider',provider.get('label','X'))
        persist(result)
        return result
    finished = [r for r in records if reusable(r)]
    pending = [r for r in records if not reusable(r)]
    with tqdm(total=len(records), initial=len(finished), desc=f'{name} judge',
              unit='question', dynamic_ncols=True) as progress:
        def judge_with_progress(record):
            result = judge(record)
            progress.update(1)
            return result
        with ThreadPoolExecutor(max_workers=args.extra_judge_concurrency) as pool:
            futures = [pool.submit(judge_with_progress, r) for r in pending]
            for future in as_completed(futures):
                finished.append(future.result())
    records = finished
    # Failed API calls/parses raise before summary publication; cached successes resume.
    order={s['id']:i for i,s in enumerate(data)}
    records.sort(key=lambda r:order[r['index']])
    for r in records:
        if r.get('judge_status') == 'ok':
            r.setdefault('judge_provider', config.get('label', 'X'))
            r.setdefault('judge_model', config['model'])
            r.setdefault('judge_endpoint', config['base_url'])
    grouped=defaultdict(list)
    for r in records: grouped[r['category']].append(r['correct'])
    summary=dict(n=len(records),correct=sum(r['correct'] for r in records),
        accuracy=sum(r['correct'] for r in records)/len(records),
        by_category={k:sum(v)/len(v) for k,v in grouped.items()},
        judge_model=config['model'],judge_endpoint=config['base_url'],judge_mode='full',
        judge_fallback_model=config.get('fallback',{}).get('model'),
        judge_model_counts={m:sum(r.get('judge_model')==m for r in records)
                            for m in sorted({r['judge_model'] for r in records if r.get('judge_model')})},
        metric_profile=SCORING_VERSION,judge_errors=0,dataset_metadata=metadata,
        protocol='single greedy pass; project prompt wrapper; configured judge with optional fallback; original image resolution',
        sample_cache=str(cache),evaluation_signature=completion_signature(args,name))
    return vclient.add_boxed_summary(summary,records,args),records


def completion_signature(args, name):
    config=judge_config(args)
    value=dict(version=SCORING_VERSION,model_path=str(Path(args.model_path).resolve()),
               judge_protocol='auto_pool_o3_extraction_v1',
               metadata=args.extra_benchmark_metadata[name],judge=config['model'],
               endpoint=config['base_url'],tokens=args.max_new_tokens,min_tokens=args.min_tokens,
               placeholder=args.mm_text_placeholder,judge_tokens=args.extra_judge_max_tokens)
    if config.get('fallback'):
        value['judge_fallback'] = {k:config['fallback'][k] for k in ('model','base_url')}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def cache_has_aborted_generations(path):
    if not path or not Path(path).is_file():
        return False
    latest = {}
    for line in Path(path).read_text().splitlines():
        try:
            record = json.loads(line)
            latest[record['index']] = record
        except json.JSONDecodeError:
            continue
    return any(r.get('finish_reason') == 'abort' for r in latest.values())
