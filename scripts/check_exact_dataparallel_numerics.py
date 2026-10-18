"""Bounded, immutable-checkpoint numerical diagnostic; requires both GPU leases.

Uses production PPO loss, clipping, Adam state and unchanged CUDA defaults.
Repeated single-GPU controls distinguish numerical variability from partitioning.
No models are saved. Root must coordinate the exclusive two-device window.
"""
import argparse
import copy
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from scripts import train_exact_policy as trainer
from src.exact_policy import load_exact_policy
from src.history_policy import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--games', type=int, default=2048)
    parser.add_argument('--max-seconds', type=int, default=60)
    args = parser.parse_args()
    if torch.cuda.device_count() != 2 or not 10 <= args.max_seconds <= 90:
        parser.error('Requires exclusive lease on exactly two GPUs and 10-90 second limit')
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(args.max_seconds)
    torch.set_num_threads(1)
    started = time.monotonic()
    digest = lambda: hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    source_hash = digest()
    initial, metadata = load_exact_policy(args.checkpoint, 'cpu')
    recipe = metadata['recipe']
    saved_optimizer = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['training_state']['optimizer']
    teacher, _ = load_policy(args.reference, 'cuda:0')
    rollout_model = copy.deepcopy(initial).to('cuda:0').eval()
    torch.manual_seed(2100001931)
    torch.cuda.manual_seed_all(2100001931)
    data, rollout_metrics = trainer.rollout(rollout_model, 'cuda:0', args.games, 3900030001,
                                            opponents=(teacher, copy.deepcopy(rollout_model)))
    del teacher, rollout_model
    assert len(data['action']) >= 1024
    batch = {key: value[:1024].clone() for key, value in data.items()}
    report = {'checkpoint_sha256': source_hash, 'recipe': recipe, 'rollout': rollout_metrics,
              'precision': torch.get_float32_matmul_precision(),
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'cudnn_deterministic': torch.backends.cudnn.deterministic,
              'cudnn_benchmark': torch.backends.cudnn.benchmark,
              'global_batch_size': 1024, 'checkpoint_writes': False, 'comparisons': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['checkpoint_unchanged'] = digest() == source_hash
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    def run(dataset, settings, parallel, gradients):
        raw = copy.deepcopy(initial).to('cuda:0').eval()
        optimizer = trainer.optimizer_for(raw, settings)
        optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        torch.manual_seed(991771)
        torch.cuda.manual_seed_all(991771)
        captured = []
        original_clip = torch.nn.utils.clip_grad_norm_
        def capture(parameters, *a, **kw):
            parameters = list(parameters)
            captured.append(torch.cat([p.grad.detach().flatten() for p in parameters if p.grad is not None]).cpu())
            return original_clip(parameters, *a, **kw)
        if gradients:
            torch.nn.utils.clip_grad_norm_ = capture
        try:
            metrics = trainer.update(raw, optimizer, dataset, 'cuda:0', settings,
                                     parallel_update_devices=[0, 1] if parallel else None)
            for gpu in (0, 1):
                torch.cuda.synchronize(gpu)
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
        weights = torch.cat([p.detach().flatten().cpu() for p in raw.parameters()])
        assert torch.isfinite(weights).all()
        return {'metrics': metrics, 'weights': weights, 'gradient': captured[0] if captured else None}

    def compare(left, right):
        result = {'parameter_max_abs': float((left['weights'] - right['weights']).abs().max()),
                  'same_updates': left['metrics']['updates'] == right['metrics']['updates'],
                  'same_kl_stop': left['metrics']['kl_early_stop'] == right['metrics']['kl_early_stop'],
                  'metrics_absolute_difference': {key: abs(left['metrics'][key] - right['metrics'][key])
                    for key in ('loss', 'policy_loss', 'value_loss', 'approximate_kl', 'gradient_norm')},
                  'left_metrics': left['metrics'], 'right_metrics': right['metrics']}
        if left['gradient'] is not None:
            a, b = left['gradient'].double(), right['gradient'].double()
            result.update(gradient_max_abs=float((a-b).abs().max()),
                          gradient_relative_l2=float((a-b).norm() / a.norm().clamp_min(1e-30)),
                          gradient_l2=float(a.norm()), finite_gradients=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
        return result

    single_batch_recipe = {**recipe, 'epochs': 1, 'batch_size': 1024}
    one = run(batch, single_batch_recipe, False, True)
    repeat = run(batch, single_batch_recipe, False, True)
    parallel = run(batch, single_batch_recipe, True, True)
    assert one['metrics']['updates'] == repeat['metrics']['updates'] == parallel['metrics']['updates'] == 1
    report['comparisons']['one_batch_single_repeat'] = compare(one, repeat)
    report['comparisons']['one_batch_parallel'] = compare(one, parallel)
    save()
    if time.monotonic() - started < args.max_seconds - 18:
        first = run(data, recipe, False, False)
        second = run(data, recipe, False, False)
        report['comparisons']['full_update_single_repeat'] = compare(first, second)
        save()
    signal.alarm(0)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
