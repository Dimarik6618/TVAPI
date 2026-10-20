"""Check the same PyTorch runtime used by the application; no models are downloaded."""
from __future__ import annotations

import argparse
import platform

from hardware import Runtime, torch_details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    print('Python:', platform.python_version(), platform.architecture()[0])
    try:
        import torch
    except ImportError:
        print('PyTorch is not installed in this environment.')
        return 1
    except (OSError, RuntimeError) as exc:
        print('PyTorch could not be loaded:', exc)
        return 1
    for line in torch_details(torch):
        print(line)
    runtime = Runtime.detect()
    # select_torch executes a real CUDA allocation and matrix multiplication;
    # a listed GPU alone does not prove usable kernels/driver compatibility.
    runtime.select_torch(torch)
    print('Inference device:', runtime.device)
    for note in runtime.notes:
        print(note)
    if runtime.device.startswith('cuda'):
        print('CUDA computation: PASS')
        return 0
    print('CUDA computation: not available; analysis uses CPU.')
    if not getattr(getattr(torch, 'version', None), 'cuda', None):
        print('This PyTorch build has no CUDA runtime. For NVIDIA use Install_NVIDIA_RTX50.bat.')
    print('Check the NVIDIA driver and available GPU memory if CUDA is installed but the test fails.')
    return 2 if args.require_cuda else 0


if __name__ == '__main__':
    raise SystemExit(main())
