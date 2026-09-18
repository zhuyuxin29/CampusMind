"""python -m evaluation.run_evaluation --check-only 或 --run [--limit 3]。"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_dataset(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("cases"), list) or not data['cases']:
        raise ValueError("评估集格式或版本无效。")
    seen = set()
    for case in data['cases']:
        identifier = case.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("题目 ID 为空或重复。")
        seen.add(identifier)
        if not isinstance(case.get('question'), str):
            raise ValueError(f"{identifier} 的问题不是文本。")
        if case.get('expected_status') not in {'answered', 'insufficient', 'needs_clarification', 'invalid_input'}:
            raise ValueError(f"{identifier} 的预期状态无效。")
        if not isinstance(case.get('answer_points'), list) or not all(isinstance(p, str) for p in case['answer_points']):
            raise ValueError(f"{identifier} 的答案要点格式无效。")
        if not isinstance(case.get('evidence'), list) or (case['expected_status'] == 'answered' and not case['evidence']):
            raise ValueError(f"{identifier} 缺少有效依据。")
        if not isinstance(case.get('history'), list):
            raise ValueError(f"{identifier} 的历史格式无效。")
        for turn in case['history']:
            if not isinstance(turn, dict) or not all(isinstance(turn.get(k), str) and turn[k].strip() for k in ('question', 'answer')):
                raise ValueError(f"{identifier} 的历史轮次无效。")
        if type(case.get('human_verified')) is not bool:
            raise ValueError(f"{identifier} 缺少人工核对标记。")
    return data


def validate_evidence(data: dict, index_root: Path) -> None:
    collection = data.get('collection_name', '')
    import re
    if not isinstance(collection, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,61}[A-Za-z0-9]', collection):
        raise ValueError('评估集中的集合名称无效。')
    folder = index_root / collection
    raw = (folder / 'parents.json').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads((folder / 'index_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('state') != 'ready' or manifest.get('collection_name') != collection:
        raise ValueError('绑定的知识库集合尚未就绪。')
    if digest != data.get('parents_sha256') or digest != manifest['identity']['parents_sha256']:
        raise ValueError('父级快照版本与评估集不一致，请重新核对题目依据。')
    parents = {p['metadata']['parent_id']: p for p in json.loads(raw.decode('utf-8'))}
    for case in data['cases']:
        for evidence in case['evidence']:
            parent = parents.get(evidence.get('parent_id'))
            if parent is None or not evidence.get('quote') or evidence['quote'] not in parent['page_content']:
                raise ValueError(f"{case['id']} 的引用片段未在父级快照中找到。")
            if any(evidence.get(k) != parent['metadata'].get(k) for k in ('file_name', 'page_start', 'page_end', 'section_title')):
                raise ValueError(f"{case['id']} 的来源或页码不匹配。")


def score_result(case: dict, result: dict) -> dict:
    expected = {e['parent_id'] for e in case['evidence']}
    cited = {s.get('metadata', {}).get('parent_id') for s in result.get('sources', [])}
    return {
        'status_match': result.get('status') == case['expected_status'],
        # 仅衡量返回引用中是否包含预设依据，不是检索召回率或答案正确率。
        'reference_parent_coverage': len(expected & cited) / len(expected) if expected else None,
        'manual_review': {'correctness': None, 'completeness': None, 'citation_support': None, 'notes': ''},
    }


def run_cases(data, cases, output_dir, ask_fn, session_factory, settings):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    run_dir.mkdir()
    (run_dir / 'questions.json').write_text(json.dumps({**data, 'cases': cases}, ensure_ascii=False, indent=2), encoding='utf-8')
    (run_dir / 'settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    records = []
    with (run_dir / 'results.jsonl').open('w', encoding='utf-8') as file:
        for case in cases:
            session = session_factory()
            for turn in case['history']:
                session.add_turn(turn['question'], turn['answer'])
            try:
                result = ask_fn(case['question'], session)
            except Exception as exc:
                result = {'status': 'error', 'answer': '', 'sources': [], 'error_type': type(exc).__name__}
            record = {'id': case['id'], 'question': case['question'], 'expected_status': case['expected_status'],
                      'answer_points': case['answer_points'], 'evidence': case['evidence'],
                      'human_verified': case['human_verified'], 'result': result, **score_result(case, result)}
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
            file.flush()  # 中断时保留已完成的题目。
            records.append(record)
            print(f"{case['id']}：{result['status']}", flush=True)
    coverage = [r['reference_parent_coverage'] for r in records if r['reference_parent_coverage'] is not None]
    durations = [r['result'].get('timings_ms', {}).get('total') for r in records]
    durations = [v for v in durations if isinstance(v, (int, float))]
    summary = {
        'count': len(records), 'status_match_rate': sum(r['status_match'] for r in records) / len(records) if records else None,
        'mean_reference_parent_coverage': sum(coverage) / len(coverage) if coverage else None,
        'error_count': sum(r['result']['status'] == 'error' for r in records),
        'mean_total_ms': sum(durations) / len(durations) if durations else None,
        'human_verified_case_count': sum(r['human_verified'] for r in records),
        'note': '状态匹配和预设依据覆盖仅为辅助指标；答案正确性、完整性与引用支持须人工评阅。首题可能包含初始化耗时。',
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    def safe_cell(value):
        text = str(value)
        return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) else text
    with (run_dir / 'review.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['题号', '问题', '答案要点', '实际回答', '实际状态', '正确性0-2', '完整性0-2', '引用支持0-2', '备注'])
        for r in records:
            writer.writerow([safe_cell(v) for v in (r['id'], r['question'], '；'.join(r['answer_points']), r['result'].get('answer', ''), r['result']['status'])] + ['', '', '', ''])
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description='CampusMind 评估：默认仅校验，--run 才会调用真实接口。')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--run', action='store_true', help='实际调用问答服务，可能产生接口费用')
    parser.add_argument('--questions', type=Path, default=Path(__file__).with_name('questions.json'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).with_name('results'))
    parser.add_argument('--limit', type=int, help='只运行前 N 道题')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit 必须是正整数')
    import config
    try:
        data = load_dataset(args.questions)
        validate_evidence(data, config.INDEX_ROOT)
        if data['collection_name'] != config.COLLECTION_NAME:
            raise ValueError('当前配置的集合与题集绑定集合不一致。')
        print(f"评估集校验通过：{len(data['cases'])} 题；答案要点需人工对照 PDF 核对。")
        if not args.run:
            return 0
        from campusmind.qa_service import QAService
        from campusmind.session_manager import ChatSession
        service = QAService(collection_name=data['collection_name'])
        keys = ('COLLECTION_NAME', 'DEEPSEEK_MODEL', 'EMBEDDING_MODEL', 'RERANK_MODEL', 'RECALL_K',
                'CANDIDATE_K', 'CONTEXT_TOP_N', 'MAX_CONTEXT_CHARS', 'HISTORY_ROUNDS', 'MAX_QUERIES')
        settings = {key: getattr(config, key) for key in keys}
        settings['dataset_sha256'] = hashlib.sha256(args.questions.read_bytes()).hexdigest()
        path = run_cases(data, data['cases'][:args.limit], args.output_dir, service.ask, ChatSession, settings)
        print(f'评估结果已保存：{path}')
        return 0
    except KeyboardInterrupt:
        print('已中断，已完成的题目保存在本次 results.jsonl 中。', file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'评估未完成：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
