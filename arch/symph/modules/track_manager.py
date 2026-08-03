import torch
from arch.vocab import *

class TrackManager:
    """ Used during inference to manage track IDs """
    def __init__(self, batch_size, track_num, device=None, track_id_num=META_TRACK_VOCAB_SIZE):
        self.batch_size = batch_size
        self.track_num = track_num
        self.track_id_num = track_id_num
        self.device = device

        # [Bar, B, track_num] - Stores track_ids
        self.all_track_ids: list[torch.LongTensor] = []
        # [Bar, B, track_num] - Stores instrument programs
        self.all_insts: list[torch.LongTensor] = []

        # State tracking
        self.cur_inst = torch.full((batch_size, track_id_num), META_PAD_INST, dtype=torch.long, device=device)
        self.prev_bar_track_map = torch.full((batch_size, track_id_num), -1, dtype=torch.long, device=device)
        self.cur_bar_num = 0

    def init_bar(self):
        self.bar_track_id = torch.full((self.batch_size, self.track_num), META_PAD_TRACK, device=self.device)
        self.bar_inst = torch.full((self.batch_size, self.track_num), META_PAD_INST, device=self.device)
        # Map: track_id -> tid (index in the bar)
        self.bar_track_map = torch.full((self.batch_size, self.track_id_num), -1, device=self.device)
        self.tid_counter = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

    def update_track(self, track_id, new_inst, active_batch):
        batch_id = torch.where(active_batch)[0]
        if batch_id.numel() == 0:
            return

        track_id = track_id[batch_id]

        # 1. Update Instrument state
        # Only update if the track doesn't exist in cur_inst
        no_inst_mask = (self.cur_inst[batch_id, track_id] == META_PAD_INST)
        if no_inst_mask.any():
            no_inst_batch_id = batch_id[no_inst_mask]
            self.cur_inst[no_inst_batch_id, track_id[no_inst_mask]] = new_inst[no_inst_batch_id]

        # 2. Record bar data
        tid = self.tid_counter[batch_id]
        self.bar_track_id[batch_id, tid] = track_id
        self.bar_inst[batch_id, tid] = self.cur_inst[batch_id, track_id]
        self.bar_track_map[batch_id, track_id] = tid

        self.tid_counter[batch_id] += 1

    def get_track_prev_map(self, track_id):
        batch_id = torch.arange(self.batch_size, device=self.device)
        prev_tid = self.prev_bar_track_map[batch_id, track_id]
        return prev_tid

    def update_bar(self):
        self.all_track_ids.append(self.bar_track_id)
        self.all_insts.append(self.bar_inst)
        self.prev_bar_track_map = self.bar_track_map
        self.cur_bar_num += 1

    def pack_track_data_tensor(self):
        track_id_data = torch.stack(self.all_track_ids, dim=1)
        inst_data = torch.stack(self.all_insts, dim=1)
        return track_id_data, inst_data

    def rollout_expand(self, sample_idx: list[tuple[int, int]], group_size: int):
        """ Expand the track manager for the next bar rollout:
            sample idx has list of (batch id, bar id)
            for each sample, copy the batch to group size, with the state before bar id
        """
        G = group_size
        num_samples = len(sample_idx)
        new_batch_size = num_samples * G

        new_manager = TrackManager(new_batch_size, self.track_num, device=self.device, track_id_num=self.track_id_num)

        for sid, (batch_id, bar_id) in enumerate(sample_idx):
            start_id = sid * G
            end_id = (sid + 1) * G

            new_manager.cur_inst[start_id: end_id] = self.cur_inst[batch_id].unsqueeze(0).expand(G, -1)
            new_manager.cur_bar_num = bar_id

            if bar_id > 0:
                hist_track_ids = self.all_track_ids[bar_id - 1][batch_id] # [track_num]

                # Reconstruct the map: track_id -> position_in_bar
                temp_map = torch.full((self.track_id_num,), -1, dtype=torch.long, device=self.device)

                # Find valid (non-padding) tracks from the historical bar
                valid_mask = (hist_track_ids != META_PAD_TRACK) & (hist_track_ids != META_END_OF_BAR)
                valid_tids = torch.where(valid_mask)[0]
                valid_track_ids = hist_track_ids[valid_mask]

                # Map the track_id to its index (tid) in that bar
                temp_map[valid_track_ids] = valid_tids

                # Expand this map to all members of the group
                new_manager.prev_bar_track_map[start_id: end_id] = temp_map.unsqueeze(0).expand(G, -1)

        new_manager.init_bar()
        return new_manager
