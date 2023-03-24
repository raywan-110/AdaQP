import time
import torch
import os
from contextlib import contextmanager

from ..communicator import BitType
from ..communicator import Communicator as comm


class Timer(object):

    def __init__(self, device: torch.device):
        super(Timer, self).__init__()
        self._record = {}
        self._total_record = []
        self._device = device

    @contextmanager
    def record(self, name: str):
        if name in self._record:
            raise Exception(f'{name} already exists')
        torch.cuda.current_stream(self._device).synchronize()
        start = time.time()
        yield
        torch.cuda.current_stream(self._device).synchronize()
        end = time.time()
        self._record[name] = (start, end)

    def epoch_traced_time(self):
        total_com = 0
        total_quant = 0
        total_dequant = 0
        total_central_agg = 0
        total_marginal_agg = 0
        total_full_agg = 0
        for name, (start, end) in self._record.items():
            if 'communication' in name:
                total_com += end - start
            elif 'quantization' in name and 'de' not in name:
                total_quant += end - start
            elif 'de-quantization' in name:
                total_dequant += end - start
            elif 'central' in name:
                total_central_agg += end - start
            elif 'marginal' in name:
                total_marginal_agg += end - start
            elif 'full' in name:
                total_full_agg += end - start
            else:
                raise KeyError(f'no {name} key')
        return [total_com, total_quant + total_dequant, total_central_agg, total_marginal_agg, total_full_agg]

    def clear(self, is_train: bool = True):
        # store in the _total_record for backup
        if is_train:
            self._total_record.append(self.epoch_traced_time())
        self._record = {}

    def persist(self, run: int, bit_type: BitType, exp_dir: str = 'exp'):
        store_dir = f'{exp_dir}/time_record'
        mode = 'full' if bit_type == BitType.FULL else 'quant'
        if not os.path.exists(store_dir):
            os.mkdir(store_dir)
        torch.save(self._total_record, f'{store_dir}/run{run}_{mode}worker{comm.get_rank()}_time_record.pt')

        self._total_record = []
