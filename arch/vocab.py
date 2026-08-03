from data_prep.vocab import COMMON_RANGE_NUM
from utils.vocab import Vocab

POS_NUM = COMMON_RANGE_NUM
TRACK_ID_NUM = COMMON_RANGE_NUM
INST_NUM = 129 # 128 MIDI instruments and drum
PITCH_NUM = 128
DUR_NUM = 64
QUARTER = 8

# --- Meta (Bar and Track) Vocabulary ---
META_PAD_BAR_LEN = 0
META_BAR_LEN_VOCAB_SIZE = POS_NUM

MIDI_TRACK_LIMIT = TRACK_ID_NUM - 1 # Reserve one track for harmony
META_HARMO_TRACK_ID = TRACK_ID_NUM - 1
META_END_OF_BAR = TRACK_ID_NUM
META_PAD_TRACK = TRACK_ID_NUM + 1
META_TRACK_VOCAB_SIZE = TRACK_ID_NUM + 2

META_DRUM_INST = INST_NUM - 1
META_PAD_INST = INST_NUM # In the model's vocabulary but not used in serialization
META_HARMO_INST = INST_NUM # Not in the model's vocabulary but used in serialization
META_INST_VOCAB_SIZE = INST_NUM + 1

# --- Shared Event Vocabulary --
shared_vocab = Vocab()

PAD_EVENT = shared_vocab.add_token("PAD")
END_EVENT = shared_vocab.add_token("END")
shared_vocab.add_token()

OFFSET_PITCH, is_vocab_pitch, pack_pitch, unpack_pitch, PITCH_SLICE = \
    shared_vocab.add_tokens(PITCH_NUM, "Pitch")

OFFSET_POS, is_vocab_pos, pack_pos, unpack_pos, POS_SLICE = \
    shared_vocab.add_tokens(POS_NUM, "Pos")

OFFSET_DUR, is_vocab_dur, pack_dur, unpack_dur, DUR_SLICE = \
    shared_vocab.add_tokens(DUR_NUM, "Dur", val_start=1)

OFFSET_LEGATO, is_vocab_legato, pack_legato, unpack_legato, LEGATO_SLICE = \
    shared_vocab.add_tokens(DUR_NUM, "Legato", val_start=1)
    # Legato token can save tokens when notes are following one other
