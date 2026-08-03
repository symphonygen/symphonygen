"""
Single source of truth for the conventions shared across the post-training
pipeline (rollout -> reward -> advantage -> GRPO training / final evaluation).

The stages communicate through the filesystem, so several modules must agree on
the same names and index math. Import from here instead of re-deriving them.

Directory layout of a rollout / generation root:

    <root>/
        cond_harmony_<cond_id>/
            song_<song_id>.mid      generated song (with the "Reference Outline" track)
            song_<song_id>.mp3      rendered audio (written during evaluation)
        tokens.json                 serialized songs   {cond_dir: {song_stem: tokens}}
        harmo_tokens.json           serialized harmony skeletons (rollout only)
        clamp3_cached.json          cached CLaMP 3 rewards {rel_name: reward}
        advantages.json             rewards and group-normalized advantages
        rewards.json / rewards.tsv  per-song evaluation results
        best_songs/                 best songs copied out (when pick_best_num > 0)

A "rel_name" is the path of a song relative to the root, without extension,
e.g. "cond_harmony_3/song_0".

Song-key convention: generators return a flat {song_key: bars} dict for a list
of conditions, condition-major:

    song_key = cond_id * group_size + song_id
"""

def song_key(cond_id: int, song_id: int, group_size: int) -> int:
    return cond_id * group_size + song_id

def split_song_key(key: int, group_size: int) -> tuple[int, int]:
    """ Returns (cond_id, song_id). """
    return divmod(key, group_size)

def cond_dir_name(cond_id: int) -> str:
    return f"cond_harmony_{cond_id}"

def song_file_stem(song_id: int) -> str:
    return f"song_{song_id}"

def harmo_file_stem(cond_id: int) -> str:
    return f"harmony_{cond_id}"

def song_rel_name(cond_id: int, song_id: int) -> str:
    return f"{cond_dir_name(cond_id)}/{song_file_stem(song_id)}"

def cond_of_rel_name(rel_name) -> str:
    """ The condition part of a rel_name ("cond_harmony_3/song_0" -> "cond_harmony_3"). """
    return str(rel_name).rsplit('/', 1)[0]

TOKENS_JSON = "tokens.json"
HARMO_TOKENS_JSON = "harmo_tokens.json"
CLAMP3_CACHE_JSON = "clamp3_cached.json"
ADVANTAGES_JSON = "advantages.json"
REWARDS_JSON = "rewards.json"
REWARDS_TSV = "rewards.tsv"
BEST_SONGS_DIR = "best_songs"

# A copy of a MIDI with the harmony outline track removed, rendered instead of
# the original so the outline is not audible (utils/midi2audio.py); such copies
# are excluded from evaluation (Results/evaluate.py).
NO_HARMO_SUFFIXES = ['.no_harmo', '.mid']
