
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
    return len(s) >= 4 and check_opposites(s[-1], s[-4]) and quad_partner(s) in adjacency_map


# Main function
def register_tile(key, adjacency_map):
    d = ["D", "R", "U", "L"]
    if key in adjacency_map:
        return False
    if quad_partner_in_keys(key, adjacency_map):
        print(quad_partner(key), " already exists, so ", key, " is skipped.")
        return False

    values = [key + d[0], key + d[1], key + d[2], key + d[3]]

    if len(values[0]) >= 2:  # 2.
        prev_letter_index = d.index(key[-1])
        values[prev_letter_index - 2] = values[prev_letter_index - 2][:-2]

    for i in range(4):  # 3.
        if len(values[i]) >= 3 and not check_opposites(key[-1], d[i]) and check_opposites(d[i], key[-2]):
            copy = values[i]
            values[i] = copy[:-3] + copy[-2] + copy[-3]

    for i in range(4):  # 4.
        if quad_partner_in_keys(values[i], adjacency_map) and not check_opposites(key[-1], d[i]):
            values[i] = quad_partner(values[i])

    # Add the results
    adjacency_map[key] = values.copy()
    return True
