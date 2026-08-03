from arch.vocab import *
from utils.vocab import *

# --- Harmony Event Vocabulary ---
harmo_vocab = Vocab()

harmo_vocab.use_shared_token(PAD_EVENT, "PAD")
harmo_vocab.use_shared_token(END_EVENT, "END")
START = harmo_vocab.add_token("START")
harmo_vocab.use_shared_tokens(OFFSET_PITCH, PITCH_NUM, "Pitch")
harmo_vocab.use_shared_tokens(OFFSET_POS, POS_NUM, "Pos")
harmo_vocab.use_shared_tokens(OFFSET_DUR, DUR_NUM, "Dur")
harmo_vocab.use_shared_tokens(OFFSET_LEGATO, DUR_NUM, "Legato")

OFFSET_BAR_LEN, is_vocab_bar_len, pack_bar_len, unpack_bar_len, BAR_LEN_SLICE = \
    harmo_vocab.add_tokens(POS_NUM, "BarLen", val_start=1)

harmo_vocab.size = harmo_vocab._cur_size
