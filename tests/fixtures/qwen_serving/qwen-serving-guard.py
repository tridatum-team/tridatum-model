"""Parent-only launcher for qwen-serving-harness.py under supervisor006 (qwen-recovery-003).

One finite supervised child: fresh receipt, absolute deadline = now + --seconds, pressure gate 1 before the G byte,
sampled group RSS ceiling 6 GiB, SIGTERM then SIGKILL on any stop reason. CPU llama.cpp 8ca4060 only
(DYLD_LIBRARY_PATH on its directory, LLAMA_ARG_DEVICE=none). --seconds defaults to the contract's evaluation child cap
(180 s: dev/final scenario chunks, startup and cleanup included); the explicit serving-surface measurement phase may use
up to 900 s (the previously approved service phase bound). Nothing above 900 is accepted. The harness receives
--duration <seconds> so its ready/request/shutdown window is checked against the same cap. No retry, no live deployment.

    .venv/bin/python qwen-recovery-003/qwen-serving-guard.py --receipt <supervisor receipt> --seconds 900 -- \
        --model-file <gguf> --expected-sha256 <sha> [--lora <adapter.gguf> --lora-sha256 <sha> --arms adapter] \
        --suite qwen-recovery-003/qwen-serving-suite.json --receipt <harness receipt> --server-log <log> --port <free>
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT = Path('/private/tmp/chat-root-c-005')
REPO = Path('/Users/eomgyuhyeon/사업/trid')
EVALUATION_SECONDS = 180  # contract.json runtime_authority.evaluation_child_seconds_max
SERVING_SECONDS_MAX = 900  # approved standalone serving-surface phase bound; never for fresh final evaluation


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--receipt', type=Path, required=True, help='supervisor receipt (must not exist)')
    p.add_argument('--seconds', type=int, default=EVALUATION_SECONDS,
                   help=f'supervised window; default {EVALUATION_SECONDS} (evaluator), max {SERVING_SECONDS_MAX} (serving phase)')
    p.add_argument('args', nargs=argparse.REMAINDER, help='harness arguments after --')
    a = p.parse_args()
    if a.receipt.exists():
        p.error('Use a fresh supervisor receipt')
    if not 0 < a.seconds <= SERVING_SECONDS_MAX:
        p.error(f'--seconds must be within 1..{SERVING_SECONDS_MAX}')
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    args = a.args[1:] if a.args[:1] == ['--'] else a.args
    if '--duration' in args:
        p.error('--duration is set by the guard')
    spec = importlib.util.spec_from_file_location('owned_cpu_supervisor', ROOT / 'institution-native-ranking-V/development_supervisor006.py')
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(Path.home()),
           'TMPDIR': '/private/tmp', 'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONNOUSERSITE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'OMP_NUM_THREADS': '4', 'LLAMA_ARG_DEVICE': 'none', 'LLAMA_ARG_N_GPU_LAYERS': '0',
           'LLAMA_ARG_LOG_VERBOSITY': '4',
           'DYLD_LIBRARY_PATH': str(REPO / 'runs/tools/llama.cpp-8ca4060')}
    os.chdir(REPO)
    receipt = supervisor.supervise([str(REPO / '.venv/bin/python'), '-B', str(ROOT / 'qwen-recovery-003/qwen-serving-harness.py'),
                                    *args, '--duration', str(a.seconds)],
                                   a.receipt.resolve(), absolute_deadline_unix=time.time() + a.seconds, duration=a.seconds,
                                   pressure_required=1, env=env)
    summary = {k: receipt.get(k) for k in ('harness_exit_code', 'stop_reason', 'reaped', 'cleanup_state',
                                          'cleanup_errors', 'hard_alarm_fired', 'first_soft_stop_reason', 'elapsed_seconds')}
    print(json.dumps(summary))
    if not (receipt.get('harness_exit_code') == 0 and receipt.get('reaped') is True
            and receipt.get('cleanup_state') == 'completed' and not receipt.get('cleanup_errors')
            and receipt.get('stop_reason') == 'harness_exit' and receipt.get('first_soft_stop_reason') is None
            and receipt.get('hard_alarm_fired') is False):
        raise SystemExit('serving measurement failed; no retry')


if __name__ == '__main__':
    main()
