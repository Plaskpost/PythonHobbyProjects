from collections import deque

"""
Tile Adjacency Registration System

This module contains all the logic for defining and handling which tiles are adjacent to each other.
"""


d = ["D", "R", "U", "L"]
range_4_duplicates = {"LDDR": "DLLU", "DLLU": "LDDR",
                      "RDDL": "DRRU", "DRRU": "RDDL",
                      "LUUR": "ULLD", "ULLD": "LUUR",
                      "RUUL": "URRD", "URRD": "RUUL"}
range_5_duplicates = {'DRRUUL': 'RDDDL', 'RDDLLU': 'DRRRU',
                      'DLLUUR': 'LDDDR', 'LDDRRU': 'DRRRU',
                      'RUULLD': 'URRRD', 'URRDDL': 'RUUUL',
                      'ULLDDR': 'LUUUR', 'LUURRD': 'ULLLD'}


# Main function in this context.
def register_tile(key, adjacency_map):
    """
    Main function in this context. Takes in a tile key, generates suggested names for its neighbors and adds them to the
    adjacency map.

    :param key: Provided tile key.
    :param adjacency_map: The adjacency map (dict) storing which tiles are neighbouring which.
    :returns: True if the dile was successfully registered. False otherwise.
    """
    if key not in adjacency_map:
        print_map(adjacency_map)
        raise KeyError(f"ERROR: Request to register tile {key}, despite it hasn't appeared as a neighbor.")
    if adjacency_map[key] is not None:
        return False
    if quad_partner_in_keys(key, adjacency_map):
        print(quad_partner(key), " already exists, so ", key, " is skipped.")
        return False

    # 1. Add "D", "R", "U", "L" to the current string (Down, Right, Up, Left).
    neighbors = [key + d[0], key + d[1], key + d[2], key + d[3]]  # 1.

    # 2. Any last two letters being opposites ("DU", "UD", "LR", "RL) are removed.
    if len(neighbors[0]) >= 2:
        prev_letter_index = d.index(key[-1])
        neighbors[prev_letter_index - 2] = neighbors[prev_letter_index - 2][:-2]

    # 3. Whenever the third last letter and the last letter of an element are opposites,
    # the newly added letter is removed and the last two are flipped.
    for i in range(4):
        if len(neighbors[i]) >= 3 and not check_opposites(key[-1], d[i]) and check_opposites(d[i], key[-2]):
            copy = neighbors[i]
            neighbors[i] = copy[:-3] + copy[-2] + copy[-3]


    # 4. If the last letter and the fourth last letter are opposites, the entry stays unless
    # the same string with a specific quadruple at the end is included in the list according to the following pairs:
    for i in range(4):
        c1 = quad_partner_in_keys(neighbors[i], adjacency_map)
        c2 = check_opposites(key[-1], d[i])
        if c1 and not c2:
            neighbors[i] = quad_partner(neighbors[i])

    # Add the results
    adjacency_map[key] = neighbors.copy()
    for neighbor in neighbors:
        if neighbor not in adjacency_map:
            adjacency_map[neighbor] = None
    return True


def iterative_registration(key, adjacency_map):
    """
    A slower but more careful approach to register_tile(). Iterates through the suggested key string of each neighbor to
    see if its name can be optimized further.

    :param key: Provided tile key.
    :param adjacency_map: The adjacency map (dict) storing which tiles are neighbouring which.
    :returns: True if the dile was successfully registered. False otherwise.
    """
    if key not in adjacency_map:
        print_map(adjacency_map)
        raise KeyError(f"ERROR: Request to register tile {key}, despite it hasn't appeared as a neighbor.")
    if adjacency_map[key] is not None:
        return False
    if quad_partner_in_keys(key, adjacency_map):
        print(quad_partner(key), " already exists, so ", key, " is skipped.")
        return False

    # 1. Add "D", "R", "U", "L" to the current string (Down, Right, Up, Left).
    neighbors = [key + d[0], key + d[1], key + d[2], key + d[3]]  # 1.

    # 2. Any last two letters being opposites ("DU", "UD", "LR", "RL) are removed.
    if len(neighbors[0]) >= 2:
        prev_letter_index = d.index(key[-1])
        neighbors[prev_letter_index - 2] = neighbors[prev_letter_index - 2][:-2]

    for i in range(4):
        neighbors[i] = iterative_reduction(neighbors[i], adjacency_map)

    adjacency_map[key] = neighbors
    for neighbor in neighbors:
        if neighbor not in adjacency_map:
            adjacency_map[neighbor] = None  # Add the neighbor to the keys when we've established a tile with connection to them.

    return True


def iterative_reduction(string, adjacency_map):
    """
    A sub-function to iterative_registration(). Performs the repeated reduction attempts on a given key string.

    :returns: The reduced string.
    """
    string_list = [letter for letter in string]
    no_edits = False
    while not no_edits:
        no_edits = True

        # 2. Any last two letters being opposites ("DU", "UD", "LR", "RL) are removed.
        for j in range(len(string_list) - 2, -1, -1):
            if check_opposites(string_list[j], string_list[j+1]):
                del string_list[j+1]
                del string_list[j]
                no_edits = False

        # 3. Whenever the third last letter and the last letter of an element are opposites,
        # the newly added letter is removed and the last two are flipped.
        for j in range(len(string_list) - 3, -1, -1):
            if check_opposites(string_list[j], string_list[j + 2]):
                trio_reduction(string_list, j)
                if j < len(string_list) - 2:  # If there's at least one letter ahead.
                    direction = rotation_direction(string_list[j], string_list[j + 1])
                    twist_from(string_list, j + 2, direction)
                no_edits = False
                break

        # 4. If the last letter and the fourth last letter are opposites, the entry stays unless
        # the same string with a specific quadruple at the end is included in the list.
        for j in range(len(string_list), 3, -1):
            current_prefix = ''.join(string_list[:j])
            if quad_partner_in_keys(current_prefix, adjacency_map):
                new_prefix = quad_partner(current_prefix)
                full = new_prefix + ''.join(string_list[j:])
                string_list = list(full)
                if j < len(string_list):
                    direction = rotation_direction(string_list[j-1], string_list[j])
                    twist_from(string_list, j, direction)
                no_edits = False
                break

        # 5.
        for j in range(len(string_list), 5, -1):
            current_prefix, substring_with_attention = (string_list[:(j-6)], ''.join(string_list[(j-6):j]))
            if substring_with_attention in range_5_duplicates:
                substring_with_attention = range_5_duplicates[substring_with_attention]
                string_list = current_prefix + list(substring_with_attention) + string_list[j:]
                if j < len(string_list):
                    direction = rotation_direction(string_list[j - 1], string_list[j])
                    twist_from(string_list, j, direction)
                no_edits = False
                break

    return ''.join(string_list)


def bulk_registration(adjacency_map, origin, radius):
    """
    Registers multiple tiles at a specified "radius" around the specified origin tile.

    :param adjacency_map: The adjacency map (dict) storing which tiles are neighbouring which.
    :param origin: Tile in the center of the group of tiles to be registered.
    :param radius: Tile steps away from the origin to include.
    :return:
    """
    queue = deque([(origin, 0)])

    while queue:
        current, distance = queue.popleft()

        iterative_registration(current, adjacency_map)

        if distance < radius:
            for neighbor in adjacency_map[current]:
                queue.append((neighbor, distance+1))


def link_duplicates(adjacency_map, dominant_key, recessive_key):
    """
    Assumes the two keys describe the same tile. Removes the recessive and establishes its connections with the dominant.
    """
    dominant_neighbors = adjacency_map[dominant_key]
    recessive_neighbors = adjacency_map[recessive_key]
    if dominant_neighbors is None or recessive_neighbors is None:
        raise ValueError("ERROR: Tried to link keys that are None in the adjacency_map.")

    # Identify common neighbors
    common_neighbors = set()
    index_diff = 0
    for i, neighbor_to_dominant in enumerate(dominant_neighbors):
        for j, neighbor_to_recessive in enumerate(recessive_neighbors):
            if neighbor_to_recessive == neighbor_to_dominant:
                index_diff = i - j
                common_neighbors.add(neighbor_to_dominant)

    if not common_neighbors:
        raise RuntimeError(f"ERROR: {dominant_key} and {recessive_key} claimed to describe same key but share no neighbors.")

    for j, neighbor in enumerate(recessive_neighbors):
        # Remove or ignore neighbors that aren't fully initialized.
        if adjacency_map[neighbor] is None:
            if neighbor not in common_neighbors:
                del adjacency_map[neighbor]
            continue


        # Rerouting common neighbors' adjacencies to dominant_key.
        if recessive_key in adjacency_map[neighbor]:
            index = adjacency_map[neighbor].index(recessive_key)
            adjacency_map[neighbor][index] = dominant_key

        if neighbor not in common_neighbors:  # and is neither None, it's an established key that shouldn't exist.
            dominant_index = (j + index_diff) % 4
            dominant_neighbor = dominant_neighbors[dominant_index]

            # To solve this, call the function recursively on the neighboring tile.
            link_duplicates(adjacency_map, dominant_neighbor, neighbor)

    del adjacency_map[recessive_key]
    return recessive_key


# ------------------- STRING OPERATIONS -------------------

def trio_reduction(string, start):
    """
    If any part of the key string includes a turn in the same direction twice in a row, the string can be written
    shorter. For example: the key 'ULD' turns left twice, while the key 'LU' reaches the same tile in fewer steps.

    :param string: Full string to reduce.
    :type string: list
    :param start: Index to where the double-turn three-letter segment starts.
    :return:
    """
    del string[start+2]
    copy = string[start]
    string[start] = string[start+1]
    string[start+1] = copy

def twist_from(string, index, direction):
    """
    Most reductions affect the orientation of the map relative some unit that follows the steps described by the key
    string. This function corrects the letters succeeding the reduced bloch of the string to follow this new orientation.

    :param string: Full string.
    :type string: list
    :param index: Index to the first letter in the string that should follow a new orientation.
    :param direction: Direction of the orientation change. +n for n steps counter-clockwise. -n for n steps clockwise.
    :return:
    """
    for i in range(index, len(string)):
        letter = string[i]
        string[i] = d[(d.index(letter) + direction) % 4]

def drul_to_brfl(string, initial_direction='U'):
    """
    In some situations, it may be beneficial to keep track of how a unit "turns" when traversing the grid rather than
    directions given some orientation. This function converts a key string given as "drul" = down/right/left/up to
    brfl = back/right/forward/left given some initial direction.

    :param string: full string
    :param initial_direction: Since turning direction is based on the previous letter in the string, an initial
    reference is needed to find the first entry.
    :returns: string in brfl format.
    """
    brfl = ['B', 'R', 'F', 'L']
    brfl_string = ""
    previous_direction = initial_direction
    for s in string:
        brfl_index = (d.index(s) - d.index(previous_direction) + 2) % 4
        brfl_string += brfl[brfl_index]
        previous_direction = s

    return brfl_string

def get_reversed_path_string(string):
    """
    Finds the "inverse" to a given key string = The string that describes the steps from the end tile to the origin tile.
    """
    s = ""
    for i in range(len(string)-1, -1, -1):
        s += opposite_of(string[i])

    return s


# ------------------ UTILITY FUNCTIONS --------------------

def quad_partner_in_keys(s, adjacency_map):
    """
    The first tiles that can be described by two equally long key strings are four steps away from the origin.
    These key strings I named "quad partners" (also applies to any two strings that are equal up to a pair of quad
    partners). The function checks whether the quad partner to a given string is already registered in the map.

    :param s: Key string.
    :param adjacency_map: The adjacency map (dict) storing which tiles are neighbouring which.
    :returns: True if there exists a quad partner and this one is present in the adjacency map already. False if not.
    """
    return len(s) >= 4 \
        and check_opposites(s[-1], s[-4]) \
        and s[-2] == s[-3] \
        and s[-2] != s[-1] \
        and s[-2] != s[-4] \
        and quad_partner(s) in adjacency_map

def quad_partner(s):
    """
    Finds the quad partner to a given key string, if any. Returns None if none exists. This function used to be a lot
    more complicated, but I rewrote it to a simple lookup approach and let the rest of the code be.

    :param s: Key string.
    :returns: The quad partner to s if any exists. None if not.
    """
    quad = s[-4:]
    pre_s = s[:-4]

    try:
        return pre_s + range_4_duplicates[quad]
    except KeyError:
        print("WARNING: ", s, " has no partner!")
        return None

def rotation_direction(fromm, to):
    """
    Most reductions affect the orientation of the map relative some unit that follows the steps described by the key
    string. This function finds what rotation "direction" (+-n) is required to rotate 'fromm' to 'to'.

    :param fromm: Reference letter ('D'/'R'/'U'/'L').
    :param to: Target letter.
    :returns: Direction of the orientation change. +n for n steps counter-clockwise. -n for n steps clockwise.
    """
    return (d.index(to) - d.index(fromm) + 1) % 4 - 1

def check_opposites(a, b):
    """
    Checks whether letters at variables a and b point in the opposite directions.
    """
    if a not in d or b not in d:
        raise ValueError(f"ERROR: {a} or {b} is not a valid direction.")

    shifted_a = d[d.index(a) - 2]
    if shifted_a == b:
        return True
    return False

def opposite_of(letter):
    """
    Returns the letter pointing in the opposite direction to 'letter'.
    """
    if letter not in d:
        raise ValueError(f"ERROR: {letter} is not a part of {d}.")
    return d[d.index(letter) - 2]


# -------------------- DEBUG PRINT ---------------------
def print_map(adjacency_map):
    for name, adjacencies in adjacency_map.items():
        print(f"{name} : {adjacencies}")

