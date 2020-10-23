from collections import deque, defaultdict
import numpy as np
import torch

from .distributed import all_reduce


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=10):
        self.deque = deque(maxlen=window_size)
        self.value = np.nan
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value
        self.value = value

    @property
    def median(self):
        values = np.array(self.deque)
        return np.median(values)

    @property
    def avg(self):
        values = np.array(self.deque)
        return np.mean(values)

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger:
    def __init__(self, delimiter=", "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if 'loss' in name:
                loss_str.append(
                    "{}: {:.6f} ({:.6f})".format(name, meter.avg, meter.global_avg)
                )
            else:
                loss_str.append(
                    "{}: {:.3f} ({:.3f})".format(name, meter.avg, meter.global_avg)
                )
        return self.delimiter.join(loss_str)


def update_meters(num_gpus, loss_dict, acc_dict, meters):
    # Gather all the predictions across all the devices.
    if num_gpus > 1:
        keys = []
        values = []
        for k in sorted(loss_dict.keys()):
            keys.append(k)
            values.append(loss_dict[k])
        for k in sorted(acc_dict.keys()):
            keys.append(k)
            values.append(acc_dict[k])
        reduced_values = all_reduce(values)
        reduced_dict = {k: v for k, v in zip(keys, reduced_values)}
        if len(loss_dict) == 1:
            meters.update(**reduced_dict)
        else:
            loss = sum(loss for loss in loss_dict.values())
            meters.update(loss=loss, **reduced_dict)
    else:
        if len(loss_dict) == 1:
            meters.update(**loss_dict, **acc_dict)
        else:
            loss = sum(loss for loss in loss_dict.values())
            meters.update(loss=loss, **loss_dict, **acc_dict)
