"""
Vocabulary used for data serialization.
Adapted from https://github.com/symphonynet/SymphonyNet.
"""

pit2alphabet = ['C', 'd', 'D', 'e', 'E', 'F', 'g', 'G', 'a', 'A', 'b', 'B']
char2pit = {pit: id for id, pit in enumerate(pit2alphabet)}
ord_A, ord_a, ord_Z, ord_z = ord('A'), ord('a'), ord('Z'), ord('z')
ord_0, ord_9 = ord('0'), ord('9')
COMMON_RANGE_NUM = 124

def int2char(n):
    if n <= 9:
        return str(n)
    elif n <= 35:
        return chr(ord_a + (n - 10))
    elif n < 62:
        return chr(ord_A + (n - 36))
    else:
        raise ValueError(f'invalid number: {n}')

def char2int(c):
    num = ord(c)
    if num >= ord_0 and num <= ord_9:
        return num - ord_0
    elif num >= ord_a and num <= ord_z:
        return 10 + num - ord_a
    elif num >= ord_A and num <= ord_Z:
        return 36 + num - ord_A
    else:
        raise ValueError(f'invalid character: {c}')

# --- Value to token conversion ---
def pit2str(n):
    octave = n // 12
    octave = octave - 1 if octave > 0 else 'O'
    rel_pit = n % 12
    return pit2alphabet[rel_pit] + str(octave)

def met2str(n):
    if n < 62:
        return 'm' + int2char(n)
    elif n < 124:
        return 'M' + int2char(n - 62)
    else:
        raise ValueError(f'invalid meter: {n}')

def pos2str(n):
    if n < 62:
        return 'p' + int2char(n)
    elif n < 124:
        return 'P' + int2char(n - 62)
    else:
        raise ValueError(f'invalid position: {n}')

def dur2str(n):
    if n < 62:
        return 'r' + int2char(n)
    elif n < 124:
        return 'R' + int2char(n - 62)
    else:
        raise ValueError(f'invalid duration: {n}')

def trk2str(n):
    if n < 62:
        return 't' + int2char(n)
    elif n < 124:
        return 'T' + int2char(n - 62)
    else:
        raise ValueError(f'invalid track_id: {n}')

def ins2str(n):
    if n < 62:
        return 'x' + int2char(n)
    elif n < 124:
        return 'X' + int2char(n - 62)
    elif n < 186:
        return 'y' + int2char(n - 124)
    else:
        raise ValueError(f'invalid instrument program: {n}')

# --- Token to value conversion ---
def str2pit(s: str):
    rel_pit = char2pit[s[0]]
    octave = (int(s[1]) if s[1] != 'O' else -1) + 1
    return octave * 12 + rel_pit

def str2met(s: str):
    if len(s) < 2:
        raise ValueError(f'Invalid meter string: {s}')

    prefix = s[0]
    char_val = s[1:]

    if prefix == 'm':
        return char2int(char_val)
    elif prefix == 'M':
        return char2int(char_val) + 62
    else:
        raise ValueError(f'Invalid meter prefix: {prefix}')

def str2pos(s: str):
    if len(s) < 2:
        raise ValueError(f'Invalid position string: {s}')

    prefix = s[0]
    char_val = s[1:]

    if prefix == 'p':
        return char2int(char_val)
    elif prefix == 'P':
        return char2int(char_val) + 62
    else:
        raise ValueError(f'Invalid position prefix: {prefix}')

def str2dur(s: str):
    if len(s) < 2:
        raise ValueError(f'Invalid duration string: {s}')

    prefix = s[0]
    char_val = s[1:]

    if prefix == 'r':
        return char2int(char_val)
    elif prefix == 'R':
        return char2int(char_val) + 62
    else:
        raise ValueError(f'Invalid duration prefix: {prefix}')

def str2trk(s: str):
    if len(s) < 2:
        raise ValueError(f'Invalid track_id string: {s}')

    prefix = s[0]
    char_val = s[1:]

    if prefix == 't':
        return char2int(char_val)
    elif prefix == 'T':
        return char2int(char_val) + 62
    else:
        raise ValueError(f'Invalid track_id prefix: {prefix}')

def str2ins(s: str):
    if len(s) < 2:
        raise ValueError(f'Invalid instrument program string: {s}')

    prefix = s[0]
    char_val = s[1:]

    if prefix == 'x':
        return char2int(char_val)
    elif prefix == 'X':
        return char2int(char_val) + 62
    elif prefix == 'y':
        return char2int(char_val) + 124
    else:
        raise ValueError(f'Invalid instrument program prefix: {prefix}')

# --- Token checking functions ---
def is_pitch(s: str):
    return len(s) == 2 and s[0] in char2pit and (s[1] == 'O' or s[1].isdigit())

def is_met(s: str):
    if len(s) < 2:
        return False
    prefix = s[0]
    if prefix not in ['m', 'M']:
        return False
    try:
        char2int(s[1:])
        return True
    except AssertionError:
        return False
def is_met_unsafe(s: str):
    return s == 'm' or s == 'M'

def is_pos(s: str):
    if len(s) < 2:
        return False
    prefix = s[0]
    if prefix not in ['p', 'P']:
        return False
    try:
        char2int(s[1:])
        return True
    except AssertionError:
        return False

def is_dur(s: str):
    if len(s) < 2:
        return False
    prefix = s[0]
    if prefix not in ['r', 'R']:
        return False
    try:
        char2int(s[1:])
        return True
    except AssertionError:
        return False

def is_trk(s: str):
    if len(s) < 2:
        return False
    prefix = s[0]
    if prefix not in ['t', 'T']:
        return False
    try:
        char2int(s[1:])
        return True
    except AssertionError:
        return False

def is_ins(s: str):
    if len(s) < 2:
        return False
    prefix = s[0]
    if prefix not in ['x', 'X', 'y']:
        return False
    try:
        char2int(s[1:])
        return True
    except AssertionError:
        return False
