import bisect

class OffsetIndexer:
    """
    Utility class to map a global index to a nested list structure.
    Example: local counts [10, 20] -> global offsets [10, 30]
    Index 15 -> global_id 1, local_id 5
    """
    def __init__(self, local_counts: list[int]):
        total = 0
        self.offsets = []
        for count in local_counts:
            total += count
            self.offsets.append(total)
        self.total_size = total

    def get_indices(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range for size {self.total_size}")

        # Find the first offset strictly greater than idx
        global_id = bisect.bisect_right(self.offsets, idx)

        # Calculate local index
        if global_id == 0:
            local_id = idx
        else:
            local_id = idx - self.offsets[global_id - 1]

        return global_id, local_id

    def __len__(self):
        return self.total_size
