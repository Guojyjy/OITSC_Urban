from collections import defaultdict
import random

class PrettyFloat(float):
    def __repr__(self):
        return "%0.2f" % self


class Dict(defaultdict):
    def __init__(self):
        defaultdict.__init__(self, Dict)

    def __repr__(self):
        return dict.__repr__(self)


def random_sample(count, start, stop, step=1):
    """
    return a list with non-overlapping random integers, len = count, range: [start, stop]
    """

    def gen_random():
        while True:
            yield random.randrange(start, stop, step)

    def gen_n_unique(source, n):
        seen = set()
        # seen.add always returns None, so 'not seen.add(x)' is always True,
        # but will only be called if the value is not already in seen (because
        # 'and' short-circuits)
        for i in (i for i in source() if i not in seen and not seen.add(i)):
            yield i
            if len(seen) == n:
                break

    return [i for i in gen_n_unique(gen_random,
                                    min(count, int(abs(stop - start) / abs(step))))]
