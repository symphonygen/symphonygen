import torch

class Vocab:
    def __init__(self):
        self._cur_size = 0
        self.size: int = None
        self.word2idx = {}
        self.idx2word = []

    def add_token(self, name=None):
        idx = self._cur_size
        self._cur_size += 1
        name = name or f"Unnamed {idx}"
        self.word2idx[name] = idx
        self.idx2word.append(name)
        return idx

    def add_tokens(self, num, name=None, val_start=0):
        offset = self._cur_size
        self._cur_size += num

        for i in range(num):
            word = f"Unnamed {offset + i}" if name is None else f"{name} {i}"
            self.word2idx[word] = offset + i
            self.idx2word.append(word)

        def is_func(x: int | torch.LongTensor) -> bool | torch.BoolTensor:
            return (offset <= x) & (x < offset + num)

        def pack_func(x: int | torch.LongTensor):
            """ Factory function for converting value to token """
            return x - val_start + offset

        def unpack_func(x: int | torch.LongTensor):
            """ Factory function for converting token to value """
            return x - offset + val_start

        return offset, is_func, pack_func, unpack_func, slice(offset, offset + num)

    def use_shared_token(self, token, name=None):
        if token != self._cur_size:
            raise ValueError(f"Token {token} should match the current size {self._cur_size}")
        self.add_token(name)

    def use_shared_tokens(self, offset, num, name=None):
        if offset != self._cur_size:
            raise ValueError(f"Offset {offset} should match the current size {self._cur_size}")
        self.add_tokens(num, name)

    def idx_seq2word_seq(self, idx_seq):
        return [self.idx2word[idx] for idx in idx_seq]

    def word_seq2idx_seq(self, word_seq):
        return [self.word2idx[word] for word in word_seq]
