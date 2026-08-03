from arch.config import *

if DEBUG_PRETRAIN:
    HARMO_EVENT_NUM = 64

    HARMO_LAYERS = 1
    HARMO_HIDDEN_SIZE = 10
    HARMO_HEADS = 2
else:
    HARMO_EVENT_NUM = 1536 # Covers 98% of 32-bar data

    HARMO_LAYERS = 12
    HARMO_HIDDEN_SIZE = 768
    HARMO_HEADS = 8
