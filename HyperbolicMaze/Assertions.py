import Explorer
import numpy as np
import HyperbolicGrid

D = 0
R = 1
U = 2
L = 3


# Some tests to see if the tile ID algorithm seems correct.
def test_map_strings():
    am = {"": None}
    HyperbolicGrid.bulk_registration(am, origin="", radius=4)

    s = ""
    for name, adjacencies in am.items():
        s += f"{name} : {adjacencies} \n"
    #print(s)

    assert "LUU" in am.keys()
    assert am["LUU"][R] == am["ULL"][D]


def test_string_twist():
    s = "ULLURDDRUU"
    s_list = list(s)
    HyperbolicGrid.twist_from(s_list, 4, 1)
    s = ''.join(s_list)
    assert s == "ULLUURRULL"
    s_list = list(s)
    HyperbolicGrid.twist_from(s_list, 5, -1)
    s = ''.join(s_list)
    assert s == "ULLUUDDRUU"

def test_iterative_registration():
    am = {"": None}
    key = "UUULDRR"
    reduced_key = HyperbolicGrid.iterative_reduction(key, am)
    assert reduced_key == "ULU"

    key = "RDL"
    reduced_key = HyperbolicGrid.iterative_reduction(key, am)
    assert reduced_key == "DR"

    key = "RDDDLU"
    am["DRRU"] = ["DRR", "DRRUR", "DRRUU", "RDD"]
    reduced_key = HyperbolicGrid.iterative_reduction(key, am)
    assert reduced_key == 'DRRUU'

def test_reverse_string():
    assert HyperbolicGrid.get_reversed_path_string("UR") == "LD"
    assert HyperbolicGrid.get_reversed_path_string("LDDRUUL") == "RDDLUUR"

def test_brfl_convertion():
    assert HyperbolicGrid.drul_to_brfl("UR", "U") == "FR"
    assert HyperbolicGrid.drul_to_brfl("LLUDLUUL", "D") == "RFRBRRFL"


if __name__ == '__main__':
    #adjacency_map = test_map_strings()
    #test_string_twist()
    test_iterative_registration()

