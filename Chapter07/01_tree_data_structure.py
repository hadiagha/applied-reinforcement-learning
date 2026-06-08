class RouteNode:
    def __init__(self, location, cost=0):
        self.location = location      # e.g., "Depot", "Customer A"
        self.cost = cost              # Accumulated travel cost or time
        self.children = []            # Possible next routes from here

    def add_child(self, child_node):
        self.children.append(child_node)

    def display(self, level=0):
        indent = "  " * level
        print(f"{indent}- {self.location} (Cost: {self.cost})")
        for child in self.children:
            child.display(level + 1)

# Example: building a small vehicle routing tree
root = RouteNode("Depot", cost=0)

# First-level routes
route_a = RouteNode("Route via North Zone", cost=50)
route_b = RouteNode("Route via South Zone", cost=60)
root.add_child(route_a)
root.add_child(route_b)

# Second-level decisions under each route
route_a.add_child(RouteNode("Deliver to Customer A first", cost=80))
route_a.add_child(RouteNode("Deliver to Customer B first", cost=70))
route_b.add_child(RouteNode("Deliver to Customer C first", cost=90))
route_b.add_child(RouteNode("Deliver to Customer D first", cost=85))

# Display the routing decision tree
root.display()
