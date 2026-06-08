from collections import deque

# Define the Node structure for routing decisions
class RouteNode:
    def __init__(self, route, cost):
        self.route = route        #A route = list of visited nodes
        self.cost = cost          #B cost = total distance or travel time
        self.children = []        #C children = next possible deliveries

    def add_child(self, child_node):
        self.children.append(child_node)


# Example vehicle routing problem using BFS and DFS
import random
import itertools

NODES = ["Depot"] + [f"C{i}" for i in range(1, 10)]  # Depot + 9 customers
distances = { (a, b): random.randint(5, 20) for a in NODES for b in NODES if a != b } # Random distances between nodes

# Build root node
root = RouteNode(route=["Depot"], cost=0)

# Initialize BFS queue
queue = deque([root])  #Initialize queue for BFS

best_route = None
best_cost = float('inf')

# Breadth-first traversal
while queue:
    node = queue.popleft()  #E Visit nodes level by level

    # If route covers all customers, check if best
    if len(node.route) == len(NODES):
        if node.cost < best_cost:
            best_cost = node.cost
            best_route = node.route
        continue

    # Expand node by adding next possible customer
    for customer in NODES:
        if customer not in node.route:
            next_cost = node.cost + distances[(node.route[-1], customer)]  # Compute incremental travel cost
            child = RouteNode(route=node.route + [customer], cost=next_cost)
            node.add_child(child)
            queue.append(child)  #Add next routes (children) to queue for next depth




print("Best route (BFS):", best_route)
print("Total cost (BFS):", best_cost)



# Define DFS as a recursive function
best_route = None
best_cost = float('inf')

def dfs(node):
    global best_route, best_cost

    # Check if route is complete: all customers visited
    if len(node.route) == len(NODES):
        if node.cost < best_cost:
            best_cost = node.cost
            best_route = node.route
        return

    # Recursive exploration: add next possible customers to route
    for customer in NODES:
        if customer not in node.route:
            next_cost = node.cost + distances[(node.route[-1], customer)]  #Compute cumulative cost as we go deeper
            child = RouteNode(route=node.route + [customer], cost=next_cost)
            node.add_child(child)
            dfs(child)  #Recursively explore the child node (depth-first)




# Run DFS from the root
root = RouteNode(route=["Depot"], cost=0)
dfs(root)

print("Best route (DFS):", best_route)
print("Total cost:", best_cost)

