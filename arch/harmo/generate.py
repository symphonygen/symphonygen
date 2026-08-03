import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from arch.config import *
from arch.harmo.config import HARMO_EVENT_NUM
from arch.harmo.sampling import *
from arch.harmo.vocab import *
from arch.symph.sampling import sample_from_logits
from data_prep.data_structure import MusicNote
from data_prep.harmo_analysis import HarmonySkeletonAnalyzer
from rl.config import MAX_HARMO_BATCH_SIZE

def generate_harmony(
    decoder: GPT2LMHeadModel,
    batch_size = 1,
    bar_num = BAR_NUM,
    event_num = HARMO_EVENT_NUM,
    start_chords_filter: list[str] | None = None,
):
    B = batch_size
    device = next(decoder.parameters()).device

    # Input
    active_idx = torch.arange(B, device=device)
    this_event = torch.full((B,), START, dtype=torch.long, device=device)

    # Constraint
    bar_len_constrainer = BarLenConstrainer(active_idx)
    bar_count_constrainer = BarCountConstrainer(active_idx, bar_num)
    bar_empty_constrainer = BarEmptyConstrainer(active_idx)
    pos_constrainer = HarmPosConstrainer(active_idx)
    constrainers: list[SamplingConstrainer] = [
        bar_len_constrainer, bar_count_constrainer, bar_empty_constrainer, pos_constrainer
    ]

    # Output
    events = torch.full((B, event_num), PAD_EVENT, dtype=torch.long, device=device)
    events[:, 0] = START

    pbar = tqdm(range(1, event_num), desc=f"Generating {batch_size} harmony")
    for eid in pbar:
        outputs = decoder.forward(
            input_ids=this_event.unsqueeze(-1),
            past_key_values=None if eid == 1 else outputs.past_key_values, use_cache=True,
        )
        logits = outputs.logits[:, -1]

        for constr in constrainers:
            constr.apply_(logits)

        this_event = sample_from_logits(logits, top_p=0.99)

        for constr in constrainers:
            constr.update_(this_event)

        events[active_idx, eid] = this_event

        # NOTE: Two END_EVENT signal the end, first for end of bar, second for end of sequence
        finished = (events[active_idx, eid - 1] == END_EVENT) & (this_event == END_EVENT)
        if finished.any():
            keep = ~finished
            active_idx = active_idx[keep]
            this_event = this_event[keep]
            outputs.past_key_values.batch_select_indices(keep)
            for constr in constrainers:
                constr.batch_select_indices(keep)
            if len(active_idx) == 0:
                break

        # Only continue generating sequences that have the requested start chords
        if start_chords_filter and (bar_count_constrainer.bar_counts >= 2).all():
            if len(active_idx) < B:
                raise RuntimeError("Should not happen")

            keep = filter_start_chord(events, start_chords_filter)
            events[~keep, 1:] = PAD_EVENT
            active_idx = active_idx[keep]
            this_event = this_event[keep]
            outputs.past_key_values.batch_select_indices(keep)
            for constr in constrainers:
                constr.batch_select_indices(keep)
            if len(active_idx) == 0:
                break

            start_chords_filter = None
            pbar.desc = f"Generating {len(active_idx)} harmony"

    if DEBUG_GEN:
        debug_events(events[0])

    return events

def filter_start_chord(events: torch.LongTensor, start_chords: list[str]):
    B = len(events)
    harmo_analyzer = HarmonySkeletonAnalyzer(triad_match_strict=True)
    has_requested = []
    for bar_events in events.tolist():
        start_chord_pitches = []
        for event in bar_events:
            if is_vocab_pitch(event):
                start_chord_pitches.append(unpack_pitch(event))
            elif start_chord_pitches:
                break
        notes = [MusicNote(0, 1, pitch) for pitch in start_chord_pitches]
        _, chord_type = harmo_analyzer.match_chord_template(notes)
        has_requested.append(chord_type in start_chords)

    print(f"{sum(has_requested)} of {B} harmonies have requested start chords")
    if sum(has_requested) > MAX_HARMO_BATCH_SIZE:
        print(f"Reducing to {MAX_HARMO_BATCH_SIZE=} to fit in GPU")
        indices = [i for i, val in enumerate(has_requested) if val][:MAX_HARMO_BATCH_SIZE]
        has_requested = [False] * B
        for idx in indices:
            has_requested[idx] = True

    return torch.tensor(has_requested, dtype=torch.bool, device=events.device)

def debug_events(events: torch.LongTensor):
    events = events.tolist()
    words = harmo_vocab.idx_seq2word_seq(events)
    with open("tmp.txt", "w") as f:
        f.write("\n".join(words))
    print("Debug: generated words saved in tmp.txt")
    __import__("pdb").set_trace()
