"""Summarize fixed three-hour developmental comparisons by independent deal."""
import hashlib
import json
from pathlib import Path

import numpy as np

base = Path(__file__).resolve().parent
directory = base / 'threehour-evaluation'
for name, expected in json.loads((directory / 'retrieval-sha256.json').read_text()).items():
    assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
captures = json.loads((directory / 'checkpoint-metadata.json').read_text())
pre_migration = json.loads((base / '5090-final-training-metadata.json').read_text())


def stats(values):
    mean = float(values.mean())
    half = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
    return dict(score=mean, paired95ci=[mean-half, mean+half])


def read(name):
    report = json.loads((directory / (name + '.json')).read_text())
    assert report['status'] == 'complete' and report['parameters_and_buffers_unchanged']
    return report


def paired(report, label):
    evaluation = next(e for e in report['evaluations'] if e['label'] == label)
    values = np.array(evaluation['outcomes']).reshape(-1, 2).mean(1)
    plan = next(o for o in report['plan']['opponents'] if o['label'] == label)
    return values, (plan['seed'], plan['games'], plan['sha256'])


summary = dict(scope='Developmental comparison after approximately three cumulative GPU hours each, '
                    'inherited 5090 budgets differ; not equal FLOPs, one seed per contender. '
                    'Shared teacher training/distillation separate. RNaD includes threshold correction.',
               candidates={}, paired_common_opponent_differences=[], rnad_sampled=[],
               promotion='None; independent final confirmation still required.')
reports = {}
for contender, capture in captures.items():
    weights = base.parents[1] / 'artifacts/checkpoints/exact-threehour' / capture['file']
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == capture['sha256']
    report = read(contender + '_greedy')
    reports[contender] = report
    total = capture['actual_training_seconds']
    earlier = pre_migration[contender]['training_seconds']
    summary['candidates'][contender] = dict(
        sha256=capture['sha256'], training_seconds=total, seconds_5090=earlier,
        seconds_4090=total-earlier, selfplay_games=capture['metadata']['games'],
        allocated_training_cost_usd=earlier/3600*.7366666667+(total-earlier)/3600*1.0585185185/2,
        results=[{k:e[k] for k in ['label', 'games', 'score', 'paired_score_95ci']}
                 for e in report['evaluations']],
    )
for label in ['validated_cnn_2h', 'previous_cnn_1h_anchor']:
    cnn, cnn_key = paired(reports['cnn_ppo'], label)
    transformer, transformer_key = paired(reports['transformer_ppo'], label)
    assert cnn_key == transformer_key
    summary['paired_common_opponent_differences'].append(
        dict(label=label, cnn_minus_transformer=stats(cnn-transformer)))
    first, first_key = paired(read('cnn_rnad_sample1'), label)
    second, second_key = paired(read('cnn_rnad_sample2'), label)
    assert first_key == second_key
    summary['rnad_sampled'].append(dict(
        label=label, independent_deals=len(first), action_replicates=2,
        games=len(first)*4, **stats((first+second)/2)))
summary['cost_caveat'] = 'Allocated training device cost only; excludes shared pretraining, tuning, evaluation, idle time and bandwidth. Not an invoice.'
(base / 'threehour-comparison-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
