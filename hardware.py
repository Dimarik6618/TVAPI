"""Probe real compute/encoder availability and adapt resource use without frame skipping."""
from __future__ import annotations
import ctypes
import os
import platform
import sys
from dataclasses import dataclass, field
import task_control as tasks


def memory_info():
    try:
        if sys.platform == 'win32':
            class Memory(ctypes.Structure):
                _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [
                    (name,ctypes.c_ulonglong) for name in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            info = Memory(); info.length = ctypes.sizeof(info)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):
                return info.total, info.available
        if sys.platform.startswith('linux'):
            values = {}
            with open('/proc/meminfo', encoding='ascii') as stream:
                for line in stream:
                    key, value = line.split(':',1)
                    values[key] = int(value.split()[0])*1024
            return values['MemTotal'], values.get('MemAvailable',values['MemFree'])
        total = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
        return total, total//2
    except (OSError, ValueError, AttributeError):
        return 0, 2*1024**3


def torch_details(torch):
    """Report the installed runtime, rather than inferring CUDA from the GPU name."""
    cuda_version = getattr(getattr(torch, 'version', None), 'cuda', None)
    details = [f'PyTorch: {torch.__version__}',
               'CUDA in PyTorch: ' + (str(cuda_version) if cuda_version else 'not included')]
    try:
        available = torch.cuda.is_available()
        details.append(f'CUDA available: {available}')
        if available:
            details.append('GPU: ' + torch.cuda.get_device_name(0))
            capability = torch.cuda.get_device_capability(0)
            details.append(f'GPU compute capability: {capability[0]}.{capability[1]}')
            details.append('PyTorch GPU architectures: ' + ', '.join(torch.cuda.get_arch_list()))
    except Exception as exc:
        details.append('CUDA diagnostics: ' + str(exc))
    return details


@dataclass
class Runtime:
    profile: str = 'auto'
    threads: int = 4
    batch: int = 2
    device: str = 'cpu'
    gpu: str = 'не используется'
    total_ram: int = 0
    available_ram: int = 0
    encoder: str = 'libx264'
    notes: list[str] = field(default_factory=list)
    probed: bool = False

    @classmethod
    def detect(cls, profile='auto'):
        total, free = memory_info()
        cpus = os.cpu_count() or 2
        limit = 2 if profile == 'economy' else (12 if profile == 'fast' else 6)
        threads = max(1,min(cpus,limit,max(1,int(free/(700*1024**2)))))
        return cls(profile=profile,threads=threads,batch=1 if profile=='economy' else 2,
                   total_ram=total,available_ram=free)

    def select_torch(self, torch):
        if self.probed:
            return
        self.probed = True
        torch.set_num_threads(self.threads)
        try:
            tasks.check()
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                if free < 2*1024**3:
                    self.notes.append('Мало свободной видеопамяти; ИИ работает на CPU.')
                    return
                device = 'cuda:0'
                # Real allocation + computation detects broken/incompatible drivers.
                with torch.inference_mode():
                    value = torch.ones((256,256),device=device)
                    (value @ value).sum().item()
                del value
                self.device = device
                self.gpu = torch.cuda.get_device_name(0)
                self.batch = 2 if self.profile=='economy' else min(
                    16 if self.profile=='fast' else 8,max(2,int(free/(700*1024**2))))
                torch.cuda.empty_cache()
            elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
                with torch.inference_mode():
                    torch.ones((128,128),device='mps').sum().item()
                self.device, self.gpu = 'mps', 'Apple Metal'
        except Exception as exc:
            self.device = 'cpu'
            self.notes.append(f'Проверка GPU не прошла: {exc}; используется CPU.')

    def cpu_fallback(self, torch):
        self.device, self.batch = 'cpu', 1
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.notes.append('Ошибка/нехватка памяти GPU: ИИ переключён на CPU.')

    def codec_args(self):
        if self.encoder == 'h264_nvenc':
            return ['-c:v',self.encoder,'-preset','p4','-cq','23','-b:v','0']
        if self.encoder == 'h264_qsv':
            return ['-c:v',self.encoder,'-global_quality','23']
        if self.encoder == 'h264_amf':
            return ['-c:v',self.encoder,'-quality','balanced','-rc','cqp','-qp_i','23','-qp_p','23']
        if self.encoder == 'h264_videotoolbox':
            return ['-c:v',self.encoder,'-b:v','6M']
        return ['-c:v','libx264','-preset','fast','-crf','23','-threads',str(self.threads)]

    def select_encoder(self, ffmpeg):
        listing = tasks.run([ffmpeg,'-hide_banner','-encoders'],capture_output=True)
        available = listing.stdout.decode('utf-8','replace')
        candidates = (['h264_videotoolbox'] if sys.platform=='darwin' else
                      ['h264_nvenc','h264_qsv','h264_amf'])
        for codec in candidates:
            tasks.check()
            if codec not in available:
                continue
            self.encoder = codec
            try:
                probe = tasks.run([ffmpeg,'-v','error','-f','lavfi','-i',
                                   'color=c=black:s=720x1280:r=24','-frames:v','4',
                                   '-pix_fmt','yuv420p',*self.codec_args(),'-f','null','-'],
                                  capture_output=True,timeout=12)
                if probe.returncode == 0:
                    return
            except Exception:
                pass
        self.encoder = 'libx264'

    def summary(self):
        ram = f'{self.total_ram/1024**3:.0f} ГБ RAM' if self.total_ram else 'RAM: авто'
        return (f'{platform.processor() or platform.machine()} · {ram} · '
                f'{self.threads} потоков · ИИ: {self.device} · экспорт: {self.encoder}')
