from collections import MutableMapping


class HistoryNamespace(MutableMapping):
    def __init__(self):
        self.ns = {}

    def __getitem__(self, key):
        return self.ns[key][-1]  # Rule 1. We return last value in history

    def __delitem__(self, key):
        self.ns[key].append(None)  # Rule 4. Instead of delete we will insert None in history

    def __setitem__(self, key, value):  # Rule 3. Instead of update we insert value in history
        if key in self.ns:
            self.ns[key].append(value)
        else:
            self.ns[key] = list([value, ])  # Rule 2. Instead of insert we create history list

    def __len__(self):
        return len(self.ns)

    def __iter__(self):
        return iter(self.ns)
