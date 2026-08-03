import itertools
from data_prep.data_structure import BarStruct
from data_prep.harmo_analysis import PITCH_CLASS_NAMES, HarmonySkeletonAnalyzer

def eval_harmony_cadence_num(
    bars: list[BarStruct],
    chord_match_threshold,
    return_loc=False
):
    """ How many cadences are there in the harmony track? """
    harmo_analyzer = HarmonySkeletonAnalyzer()

    chords = []
    for bid, bar in enumerate(bars):
        if bar.harmony_track is None:
            continue
        for pos, pos_group in itertools.groupby(bar.harmony_track.notes, key=lambda note: note.pos):
            (root, chord_type), score = harmo_analyzer.match_chord_template(list(pos_group), return_score=True)
            if root == "N":
                chords.append(("N", "N", (bid, pos, score)))
            else:
                chords.append((PITCH_CLASS_NAMES.index(root), chord_type, (bid, pos, score)))
            # print(f"Chord: {root} {chord_type} {score} at {bid} {pos}")

    cadence_loc = find_v_i_resolutions(chords, threshold=chord_match_threshold)
    return cadence_loc if return_loc else len(cadence_loc)

def find_v_i_resolutions(chords, threshold):
    """
    chords: list of tuples (root_index, chord_type)
    e.g., [(7, 'dom7'), (0, 'maj')] -> G7 to C
    """
    loc = []
    for i in range(len(chords) - 1):
        v_root, v_type, (bid, pos, v_score) = chords[i]
        i_root, i_type, (_, _, i_score) = chords[i + 1]
        if v_root == "N" or i_root == "N":
            continue
        if v_score < threshold or i_score < threshold:
            continue

        if v_root == get_dominant_root(i_root):
            if v_type in ['maj', 'dom7']:
                if i_type in ['maj', 'maj7', 'min', 'min7']:
                    loc.append((bid, pos, v_type, i_type, v_score, i_score))
    return loc

def get_dominant_root(tonic_root):
    return (tonic_root + 7) % 12
