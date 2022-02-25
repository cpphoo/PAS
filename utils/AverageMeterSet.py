# Specify classes or functions that will be exported

from collections import OrderedDict

__all__ = ['AverageMeter', 'AverageMeterSet']


class AverageMeterSet:
    def __init__(self, suffix=''):
        self.meters = OrderedDict()
        # suffix to be added to the string
        self.suffix = suffix

    def __getitem__(self, key):
        return self.meters[key]

    def __format__(self, format_spec):
        return ' | '.join([k + ': ' + self.meters[k].__format__(format_spec) for k in self.meters])

    def __str__(self):
        return self.__format__('.4f')

    def update(self, name, value, n=1):
        name_prefix = name + self.suffix
        if not name_prefix in self.meters:
            self.meters[name_prefix] = AverageMeter()
        self.meters[name_prefix].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        '''
        val is the average value
        n : the number of items used to calculate the average
        '''
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
