import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import check_nvidia


class Tensor:
    def __matmul__(self, other):
        return self

    def sum(self):
        return self

    def item(self):
        return 1.


def fake_torch(available=True, kernel_error=False):
    return SimpleNamespace(
        __version__='test-cu128' if available else 'test-cpu',
        version=SimpleNamespace(cuda='12.8' if available else None),
        set_num_threads=Mock(), inference_mode=contextlib.nullcontext,
        ones=Mock(side_effect=RuntimeError('no kernel image is available') if kernel_error else None,
                  return_value=Tensor()),
        cuda=SimpleNamespace(
            is_available=lambda:available, mem_get_info=lambda:(8*1024**3,8*1024**3),
            get_device_name=lambda *a:'NVIDIA RTX 5050 Laptop GPU',
            get_device_capability=lambda *a:(12,0),
            get_arch_list=lambda:['sm_120'], empty_cache=Mock()))


class HardwareDiagnosticTests(unittest.TestCase):
    def run_check(self, torch):
        output=io.StringIO()
        with patch.dict(sys.modules,{'torch':torch}), \
             patch.object(sys,'argv',['check_nvidia.py','--require-cuda']), \
             contextlib.redirect_stdout(output):
            code=check_nvidia.main()
        return code,output.getvalue()

    def test_cpu_package_is_not_reported_as_cuda_ready(self):
        torch=fake_torch(available=False)
        code,output=self.run_check(torch)
        self.assertEqual(code,2)
        self.assertIn('CUDA in PyTorch: not included',output)
        torch.ones.assert_not_called()

    def test_listed_gpu_with_failing_kernel_does_not_pass(self):
        code,output=self.run_check(fake_torch(kernel_error=True))
        self.assertEqual(code,2)
        self.assertIn('no kernel image is available',output)
        self.assertIn('Inference device: cpu',output)

    def test_success_requires_computation_on_cuda(self):
        torch=fake_torch()
        code,output=self.run_check(torch)
        self.assertEqual(code,0)
        self.assertIn('CUDA computation: PASS',output)
        self.assertEqual(torch.ones.call_args.kwargs['device'],'cuda:0')


if __name__ == '__main__':
    unittest.main()
