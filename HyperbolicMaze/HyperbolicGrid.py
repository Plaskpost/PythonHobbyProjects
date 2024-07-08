from collections import deque

def check_opposites(a, b):
    d = ["D", "R", "U", "L"]
    shifted_a = d[d.index(a) - 2]
    if shifted_a == b:
        return True
    return False


def quad_partner(s):
    quad = s[-4:]
    pre_s = s[:-4]
    if quad == "LDDR":
        return pre_s + "DLLU"
    elif quad == "DLLU":
        return pre_s + "LDDR"
    elif quad == "RDDL":
        return pre_s + "DRRU"
    elif quad == "DRRU":
        return pre_s + "RDDL"
    elif quad == "LUUR":
        return pre_s + "ULLD"
    elif quad == "ULLD":
        return pre_s + "LUUR"
    elif quad == "RUUL":
        return pre_s + "URRD"
    elif quad == "URRD":
        return pre_s + "RUUL"
    else:
        print("ERROR: ", quad, " has no partner!")
        return None


def quad_partner_in_keys(s, adjacency_map):
    return len(s) >= 4 and check_opposites(s[-1], s[-4]) and s[-2] == s[-3] and quad_partner(s) in adjacency_map and adjacency_map[quad_partner(s)] is not None


# Main function in this context
# Does at the moment not correct detoured labels.
def register_tile(key, adjacency_map):

    d = ["D", "R", "U", "L"]
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
        if quad_partner_in_keys(neighbors[i], adjacency_map) and not check_opposites(key[-1], d[i]):
            neighbors[i] = quad_partner(neighbors[i])

    # Add the results
    adjacency_map[key] = neighbors.copy()
    for neighbor in neighbors:
        if neighbor not in adjacency_map:
            adjacency_map[neighbor] = None
    return True


def iterative_registration(key, adjacency_map):
    d = ["D", "R", "U", "L"]
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
            adjacency_map[neighbor] = None
    return True


def iterative_reduction(string, adjacency_map):
    string_list = [letter for letter in string]
    no_edits = False
    while not no_edits:
        no_edits = True

        # 2. Any last two letters being opposites ("DU", "UD", "LR", "RL) are removed.
        for j in range(len(string_list) - 2, 0, -1):
            if check_opposites(string_list[j], string_list[j+1]):
                del string_list[j+1]
                del string_list[j]
                no_edits = False

        # 3. Whenever the third last letter and the last letter of an element are opposites,
        # the newly added letter is removed and the last two are flipped.
        for j in range(len(string_list) - 3, 0, -1):
            if check_opposites(string_list[j], string_list[j + 2]):
                trio_reduction(string_list, j)
                if j + 1 < len(string_list) - 1:
                    direction = rotation_direction(string_list[j], string_list[j + 1])
                    twist_from(string_list, j + 2, direction)
                no_edits = False
                break

        # 4. If the last letter and the fourth last letter are opposites, the entry stays unless
        # the same string with a specific quadruple at the end is included in the list according to the following pairs:
        for j in range(len(string_list), 4, -1):
            current_prefix = ''.join(string_list[:j])
            if quad_partner_in_keys(current_prefix, adjacency_map):
                new_prefix = quad_partner(current_prefix)
                full = new_prefix + ''.join(string_list[(j+1):])
                string_list = list(full)
                no_edits = False
                break

    return ''.join(string_list)


def bulk_registration(adjacency_map, origin, radius):
    queue = deque([(origin, 0)])

    while queue:
        current, distance = queue.popleft()

        register_tile(current, adjacency_map)

        if distance < radius:
            for neighbor in adjacency_map[current]:
                queue.append((neighbor, distance+1))


def trio_reduction(string, start):
    del string[start+2]
    copy = string[start]
    string[start] = string[start+1]
    string[start+1] = copy


def twist_from(string, index, direction):
    d = ["D", "R", "U", "L"]
    for i in range(index, len(string)):
        letter = string[i]
        string[i] = d[(d.index(letter) + direction) % 4]


def rotation_direction(fromm, to):
    d = ["D", "R", "U", "L"]
    return (d.index(to) - d.index(fromm) + 1) % 4 - 1


def print_map(adjacency_map):
    for name, adjacencies in adjacency_map.items():
        print(f"{name} : {adjacencies}")

