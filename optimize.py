import heapq
import math
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

class TerrainType(Enum):
    OPEN_SEA = 1
    COASTAL = 2
    SHALLOW = 3
    STORM = 4

class Node:
    def __init__(self, position, terrain_type):
        self.position = position
        self.terrain_type = terrain_type
        self.g = float('inf')  # Cost from start to this node
        self.h = 0  # Heuristic cost to goal
        self.f = float('inf')  # Total cost (g + h)
        self.parent = None  # To track the path

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    """Euclidean distance as heuristic function."""
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def get_neighbors(node, grid):
    """Returns valid neighbors for a node."""
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        x, y = node.position[0] + dx, node.position[1] + dy
        if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
            neighbors.append(Node((x, y), grid[x][y]))
    return neighbors

def get_risk(terrain_type):
    """Returns risk score for each terrain type."""
    risk_factor = {
        TerrainType.OPEN_SEA: 0.1,
        TerrainType.COASTAL: 0.3,
        TerrainType.SHALLOW: 0.5,
        TerrainType.STORM: 1.0  # High risk for storms
    }
    return risk_factor[terrain_type]

def cost(current, next_node, risk_threshold=0.5):
    """Cost function considering terrain type and risk."""
    base_cost = heuristic(current.position, next_node.position)
    terrain_factor = {
        TerrainType.OPEN_SEA: 1,
        TerrainType.COASTAL: 1.2,
        TerrainType.SHALLOW: 1.5,
        TerrainType.STORM: 10  # High cost for storm areas
    }
    risk_factor = get_risk(next_node.terrain_type)
    
    # Avoid areas with risk above the threshold
    if risk_factor > risk_threshold:
        return float('inf')  # Impose high cost for high-risk areas

    return base_cost * terrain_factor[next_node.terrain_type]

def a_star_with_risk(start, goal, grid, risk_threshold=0.5):
    """A* algorithm with risk consideration."""
    start_node = Node(start, grid[start[0]][start[1]])
    start_node.g = 0
    start_node.f = heuristic(start, goal)
    open_list = [start_node]
    closed_set = set()
    node_map = {start: start_node}

    while open_list:
        current = heapq.heappop(open_list)
        if current.position == goal:
            # Reconstruct the path from goal to start
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        closed_set.add(current.position)

        for neighbor in get_neighbors(current, grid):
            if neighbor.position in closed_set:
                continue

            tentative_g = current.g + cost(current, neighbor, risk_threshold)

            if tentative_g == float('inf'):  # Skip neighbors with too high risk
                continue

            if tentative_g < neighbor.g:
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = heuristic(neighbor.position, goal)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor.position not in node_map:
                    node_map[neighbor.position] = neighbor
                    heapq.heappush(open_list, neighbor)
                elif tentative_g < node_map[neighbor.position].g:
                    node_map[neighbor.position].g = tentative_g
                    heapq.heappush(open_list, node_map[neighbor.position])

    return None  # Return None if no path is found

def visualize_route(grid, start, goal, path):
    """Visualize the grid and the optimal path."""
    color_map = {
        TerrainType.OPEN_SEA: (1, 1, 1),     # White
        TerrainType.COASTAL: (0, 1, 1),      # Cyan
        TerrainType.SHALLOW: (1, 0.5, 0),    # Orange
        TerrainType.STORM: (1, 0, 0)         # Red
    }

    height, width = len(grid), len(grid[0])
    img = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            img[i, j] = color_map[grid[i][j]]

    plt.imshow(img)

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, color='black', marker='o', label='Optimal Path')

    plt.scatter(start[1], start[0], color='green', s=100, label='Start')
    plt.scatter(goal[1], goal[0], color='blue', s=100, label='Goal')

    plt.legend(loc='upper right')
    plt.title('Optimal Route Visualization')
    plt.show()

# Example grid and pathfinding call
grid = [
    [TerrainType.OPEN_SEA, TerrainType.OPEN_SEA, TerrainType.COASTAL, TerrainType.SHALLOW],
    [TerrainType.OPEN_SEA, TerrainType.STORM, TerrainType.COASTAL, TerrainType.SHALLOW],
    [TerrainType.OPEN_SEA, TerrainType.OPEN_SEA, TerrainType.OPEN_SEA, TerrainType.COASTAL],
    [TerrainType.SHALLOW, TerrainType.SHALLOW, TerrainType.COASTAL, TerrainType.COASTAL],
]

start = (0, 0)
goal = (3, 3)
risk_threshold = 0.5

optimal_path_with_risk = a_star_with_risk(start, goal, grid, risk_threshold)
if optimal_path_with_risk:
    print(f"Optimal path found: {optimal_path_with_risk}")
else:
    print("No path found within the risk threshold.")

visualize_route(grid, start, goal, optimal_path_with_risk)
