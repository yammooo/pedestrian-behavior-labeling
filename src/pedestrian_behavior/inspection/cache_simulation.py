"""Byte-bounded LRU accounting; no feature extraction, disk cache or training."""

from collections import OrderedDict


class ByteLRU:
    def __init__(self, capacity):
        if capacity is not None and (not isinstance(capacity, int) or capacity < 0):
            raise ValueError("Capacity must be nonnegative integer bytes or None")
        self.capacity = capacity
        self.entries = OrderedDict()
        self.seen = set()
        self.used = self.peak = 0
        self.reset_counts()

    def reset_counts(self):
        self.counts = dict.fromkeys(("requests", "hits", "misses", "compulsory_misses",
            "capacity_misses", "prior_epoch_hits", "requested_bytes", "miss_bytes", "read_bytes", "write_bytes",
            "evictions", "evicted_bytes", "oversize_bypasses"), 0)

    def access(self, key, size, epoch):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Entry size must be positive integer bytes")
        c = self.counts
        c["requests"] += 1
        c["requested_bytes"] += size
        if key in self.entries:
            previous_size, previous_epoch = self.entries.pop(key)
            if previous_size != size:
                raise ValueError("Immutable cache entry changed size")
            self.entries[key] = (size, epoch)
            c["hits"] += 1
            c["read_bytes"] += size
            c["prior_epoch_hits"] += previous_epoch < epoch
            return
        c["misses"] += 1
        c["miss_bytes"] += size
        c["capacity_misses" if key in self.seen else "compulsory_misses"] += 1
        self.seen.add(key)
        if self.capacity is not None and size > self.capacity:
            c["oversize_bypasses"] += 1
            return
        while self.capacity is not None and self.used+size > self.capacity:
            _, (evicted_size, _) = self.entries.popitem(last=False)
            self.used -= evicted_size
            c["evictions"] += 1
            c["evicted_bytes"] += evicted_size
        self.entries[key] = (size, epoch)
        self.used += size
        c["write_bytes"] += size
        self.peak = max(self.peak, self.used)

    def snapshot(self):
        return {**self.counts, "hit_rate": self.counts["hits"]/self.counts["requests"] if self.counts["requests"] else None,
                "resident_entries": len(self.entries), "resident_bytes": self.used,
                "peak_resident_bytes": self.peak}
