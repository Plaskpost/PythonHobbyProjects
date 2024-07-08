import math
import HyperbolicGrid

neighbors = ['LU', 'UR', 'RD', 'DL', 'UL', 'RU', 'DR', 'LD']
d = ["D", "R", "U", "L"]

for neighbor in neighbors:
    direction = HyperbolicGrid.rotation_direction(d, neighbor[0], neighbor[1])
    print(f"{''.join(neighbor)} : {direction}")


del neighbors[1]

print(neighbors)