from arch.vocab import *
from utils.vocab import *

# --- Event Vocabulary ---
event_vocab = Vocab()

event_vocab.use_shared_token(PAD_EVENT, "PAD")
event_vocab.use_shared_token(END_EVENT, "END")
END_OF_BAR = event_vocab.add_token("END_OF_BAR")
    # NOTE: END_OF_BAR event is used for the event encoder to always
    # have non-empty input, but the event decoder does not predict it,
    # as the track decoder already predicts META_END_OF_BAR.
event_vocab.use_shared_tokens(OFFSET_PITCH, PITCH_NUM, "Pitch")
event_vocab.use_shared_tokens(OFFSET_POS, POS_NUM, "Pos")
event_vocab.use_shared_tokens(OFFSET_DUR, DUR_NUM, "Dur")
event_vocab.use_shared_tokens(OFFSET_LEGATO, DUR_NUM, "Legato")

event_vocab.size = event_vocab._cur_size
