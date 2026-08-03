import mmap
import pickle
from torch.utils.data import Dataset
from arch.config import BAR_NUM, BAR_STRIDE, bar_index_path
from arch.data_tensor import DataTensorConverter
from data_prep.main import deserialize_song
from utils.offset_indexer import OffsetIndexer

class MusicDataset(Dataset):
    def __init__(
        self,
        token_path: str,
        converter: DataTensorConverter | None = None,
        bar_num=BAR_NUM, bar_stride=BAR_STRIDE,
        include_last_window=False,
    ):
        print(f"Loading {token_path}")
        self.token_path = token_path
        self.bar_num = bar_num
        self.bar_stride = bar_stride

        self.mm = self._file_obj = None

        index_path = bar_index_path(token_path)
        with open(index_path, 'rb') as f:
            self.bar_offsets = pickle.load(f)

        all_window_counts = []
        for offsets in self.bar_offsets:
            num_bars = len(offsets) - 1
            space_after_first_window = max(0, num_bars - self.bar_num)
            if include_last_window:
                # Allow the last window to be partial
                window_count = 1 + (space_after_first_window + self.bar_stride - 1) // self.bar_stride
            else:
                window_count = 1 + space_after_first_window // self.bar_stride
            # Include at least one window for each song
            # XXX: if the song is short, the window may contain trailing empty bars
            window_count = max(1, window_count)
            all_window_counts.append(window_count)

        self.indexer = OffsetIndexer(all_window_counts)
        self.converter = converter

    def _init_mmap(self):
        if self.mm is None:
            self._file_obj = open(self.token_path, 'rb')
            self.mm = mmap.mmap(self._file_obj.fileno(), 0, access=mmap.ACCESS_READ)

    def get_window(self, song_id, window_id):
        this_song_bar_offsets = self.bar_offsets[song_id]

        bar_start = window_id * self.bar_stride
        bar_end = min(bar_start + self.bar_num, len(this_song_bar_offsets) - 1)

        start_byte = this_song_bar_offsets[bar_start]
        end_byte = this_song_bar_offsets[bar_end]

        self._init_mmap()
        window_tokens = self.mm[start_byte: end_byte].decode('utf-8')
        bars = deserialize_song(window_tokens)
        return bars

    def __len__(self):
        return len(self.indexer)

    def __getitem__(self, idx):
        song_id, window_id = self.indexer.get_indices(idx)
        bars = self.get_window(song_id, window_id)
        if self.converter is None:
            return bars
        data = self.converter.pack_data_tensor(bars)
        return data

    def __del__(self):
        if self._file_obj is not None:
            self.mm.close()
            self._file_obj.close()
