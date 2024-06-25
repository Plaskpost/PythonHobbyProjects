
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
    return len(s) >= 4 and check_opposites(s[-1], s[-4]) and s[-2] == s[-3] and quad_partner(s) in adjacency_map


# Main function in this context
def register_tile(key, adjacency_map):
    d = ["D", "R", "U", "L"]
    if key in adjacency_map:
        return False
    if quad_partner_in_keys(key, adjacency_map):
        print(quad_partner(key), " already exists, so ", key, " is skipped.")
        return False

    # 1. Add "D", "R", "U", "L" to the current string (Down, Right, Up, Left).
    neighbors = [key + d[0], key + d[1], key + d[2], key + d[3]]  # 1.

    # TODO: Step 2-4 has to be repeated throughout the strings until no changes are made.
    # (do while would be nice here...)
    # ISSUE: When rule 3 are applied, all later letters are affected.

    # 2. Any last two letters being opposites ("DU", "UD", "LR", "RL) are removed.
    if len(neighbors[0]) >= 2:
        prev_letter_index = d.index(key[-1])
        neighbors[prev_letter_index - 2] = neighbors[prev_letter_index - 2][:-2]

    # 3. Whenever the third last letter and the last letter of an element are opposites,
    # the newly added letter is removed and the last two are flipped.
    for i in range(4):
        neighbors[i] = run_simplification(neighbors[i], d[i])

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
    return True


