from arch.harmo.vocab import *
from arch.base_class import SamplingConstrainer

INF = 1e9

class BarLenConstrainer(SamplingConstrainer):
    """
    Time signatures should be common (2/4, 3/4 or 4/4).
    Currently, only time signatures that have quarter-divisible beats are supported by `DissonanceConstrainer`.
    """
    allowed_bar_len = torch.tensor([16, 24, 32], dtype=torch.long)

    def __init__(self, active_idx: torch.LongTensor):
        device = active_idx.device
        self.bar_len_mask = torch.ones(POS_NUM, dtype=torch.bool, device=device).unsqueeze(0)
        self.bar_len_mask[:, self.allowed_bar_len.to(device) - 1] = False

    def apply_(self, logits: torch.Tensor):
        logits[:, BAR_LEN_SLICE].masked_fill_(self.bar_len_mask, -INF)

class BarCountConstrainer(SamplingConstrainer):
    """ Number of bars should be fixed """
    def __init__(self, active_idx: torch.LongTensor, bar_num):
        device = active_idx.device
        self.bar_num = bar_num
        self.bar_counts = torch.zeros(len(active_idx), dtype=torch.long, device=device)
        self.last_is_end = torch.zeros(len(active_idx), dtype=torch.bool, device=device)

    def apply_(self, logits):
        logits[self.last_is_end & (self.bar_counts < self.bar_num), END_EVENT] = -INF
        logits[self.last_is_end & (self.bar_counts >= self.bar_num), END_EVENT] = INF # HACK
        logits[self.bar_counts >= self.bar_num, BAR_LEN_SLICE] = -INF

    def update_(self, this_event: torch.LongTensor):
        bar_start = is_vocab_bar_len(this_event)
        self.bar_counts += bar_start.long()

        self.last_is_end = (this_event == END_EVENT)

    def batch_select_indices(self, indices: torch.LongTensor):
        self.bar_counts = self.bar_counts[indices]
        self.last_is_end = self.last_is_end[indices]

class HarmPosConstrainer(SamplingConstrainer):
    """
    Positions should not exceed bar length;
    One position only has one note group (with the same duration).
    """
    limit_at_quarter = False

    def __init__(self, active_idx: torch.LongTensor):
        device = active_idx.device

        self.pos_values = torch.arange(POS_NUM, device=device)
        self.legato_values = torch.arange(1, DUR_NUM + 1, device=device)

        self.cur_pos = torch.zeros(len(active_idx), dtype=torch.long, device=device)
        self.pos_limit = torch.zeros(len(active_idx), dtype=torch.long, device=device)

        self.last_is_dur = torch.zeros(len(active_idx), dtype=torch.bool, device=device)

    def apply_(self, logits: torch.Tensor):
        # [B, 1] vs [1, POS_NUM]
        pos_mask = (self.pos_values.unsqueeze(0) >= self.pos_limit.unsqueeze(-1))
        if self.limit_at_quarter:
            pos_mask |= (self.pos_values.unsqueeze(0) % QUARTER != 0)
        logits[:, POS_SLICE].masked_fill_(pos_mask, -INF)

        # [B, 1] vs [1, DUR_NUM]
        legato_mask = (self.legato_values.unsqueeze(0) >= (self.pos_limit - self.cur_pos).unsqueeze(-1))
        if self.limit_at_quarter:
            legato_mask |= (self.legato_values.unsqueeze(0) % QUARTER != 0)
        logits[:, LEGATO_SLICE].masked_fill_(legato_mask, -INF)

        logits[self.last_is_dur, PITCH_SLICE] = -INF # A Duration token instead of Legato means an empty beat is ahead

    def update_(self, this_event: torch.LongTensor):
        bar_start = is_vocab_bar_len(this_event)
        if bar_start.any():
            self.cur_pos[bar_start] = 0
            self.pos_limit[bar_start] = unpack_bar_len(this_event[bar_start])

        is_pos = is_vocab_pos(this_event)
        if is_pos.any():
            self.cur_pos[is_pos] = unpack_pos(this_event[is_pos])

        is_legato = is_vocab_legato(this_event)
        if is_legato.any():
            self.cur_pos[is_legato] += unpack_legato(this_event[is_legato])

        self.last_is_dur = is_vocab_dur(this_event)

    def batch_select_indices(self, indices: torch.LongTensor):
        self.cur_pos = self.cur_pos[indices]
        self.pos_limit = self.pos_limit[indices]
        self.last_is_dur = self.last_is_dur[indices]

class BarEmptyConstrainer(SamplingConstrainer):
    """
    Bar should not be empty;
    Downbeat should not be empty
    """
    def __init__(self, active_idx: torch.LongTensor):
        device = active_idx.device
        self.last_is_bar_len = torch.zeros(len(active_idx), dtype=torch.bool, device=device)

    def apply_(self, logits: torch.Tensor):
        logits[self.last_is_bar_len, END_EVENT] = -INF
        logits[self.last_is_bar_len, BAR_LEN_SLICE] = -INF
        logits[self.last_is_bar_len, POS_SLICE] = -INF

    def update_(self, this_event: torch.LongTensor):
        self.last_is_bar_len = is_vocab_bar_len(this_event)

    def batch_select_indices(self, indices: torch.LongTensor):
        self.last_is_bar_len = self.last_is_bar_len[indices]
