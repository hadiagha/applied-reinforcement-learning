import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import math
import random
from collections import defaultdict
import time

# ============================================================================
# PART 1: CVRP INSTANCE REPRESENTATION
# ============================================================================

class CVRPInstance:
    """Represents a CVRP problem instance with all necessary data."""
    
    def __init__(self, filename):
        """Load CVRP instance from file."""
        self.name = ""
        self.dimension = 0
        self.capacity = 0
        self.max_vehicles = 0
        self.coords = {}
        self.demands = {}
        self.depot = 1
        
        self._parse_file(filename)
        self._compute_distance_matrix()
        
    def _parse_file(self, filename):
        """Parse CVRP instance file."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        section = None
        for line in lines:
            line = line.strip() # Remove leading/trailing whitespace
            if not line or line == 'EOF':
                continue
                
            if line.startswith('NAME'): # Extract instance name
                self.name = line.split(':')[1].strip()
            elif line.startswith('DIMENSION'):
                self.dimension = int(line.split(':')[1].strip())
            elif line.startswith('CAPACITY'):
                self.capacity = int(line.split(':')[1].strip())
            elif line.startswith('MAX_VEHICLE'):
                self.max_vehicles = int(line.split(':')[1].strip())
            elif line == 'NODE_COORD_SECTION':
                section = 'coords'
            elif line == 'DEMAND_SECTION':
                section = 'demands'
            elif line == 'DEPOT_SECTION':
                section = 'depot'
            elif section == 'coords':
                parts = line.split()
                if len(parts) == 3:
                    node_id = int(parts[0])
                    x, y = float(parts[1]), float(parts[2])
                    self.coords[node_id] = (x, y)
            elif section == 'demands':
                parts = line.split()
                if len(parts) == 2:
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    self.demands[node_id] = demand
            elif section == 'depot':
                if line != '-1':
                    self.depot = int(line)
    
    def _compute_distance_matrix(self):
        """Precompute Euclidean distances between all nodes."""
        n = self.dimension
        self.distances = np.zeros((n + 1, n + 1))
        
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i != j:
                    x1, y1 = self.coords[i]
                    x2, y2 = self.coords[j]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    self.distances[i][j] = dist
    
    def get_distance(self, node1, node2):
        """Get distance between two nodes."""
        return self.distances[node1][node2]
    
    def get_customers(self):
        """Get list of customer nodes (excluding depot)."""
        return [i for i in range(1, self.dimension + 1) if i != self.depot]


# ============================================================================
# PART 2: CVRP STATE REPRESENTATION
# ============================================================================

class CVRPState:
    """Represents a state in the MCTS tree for CVRP."""
    
    def __init__(self, instance, routes=None, unvisited=None, current_vehicle=0):
        """
        Initialize CVRP state.
        
        Args:
            instance: CVRPInstance object
            routes: List of routes (each route is a list of customer nodes)
            unvisited: Set of unvisited customers
            current_vehicle: Index of current vehicle being constructed
        """
        self.instance = instance
        self.routes = routes if routes is not None else [[] for _ in range(instance.max_vehicles)]
        self.unvisited = unvisited if unvisited is not None else set(instance.get_customers())
        self.current_vehicle = current_vehicle
        
    def clone(self):
        """Create a deep copy of the state."""
        return CVRPState(
            self.instance,
            [route[:] for route in self.routes],
            self.unvisited.copy(),
            self.current_vehicle
        )
    
    def is_terminal(self):
        """Check if all customers are visited."""
        return len(self.unvisited) == 0
    
    def get_route_load(self, vehicle_idx):
        """Calculate current load for a vehicle route."""
        return sum(self.instance.demands[customer] for customer in self.routes[vehicle_idx])
    
    def can_add_customer(self, customer, vehicle_idx=None):
        """Check if customer can be added to current or specified vehicle."""
        if customer not in self.unvisited:
            return False
        
        if vehicle_idx is None:
            vehicle_idx = self.current_vehicle
            
        current_load = self.get_route_load(vehicle_idx)
        customer_demand = self.instance.demands[customer]
        
        return current_load + customer_demand <= self.instance.capacity
    
    def get_available_actions(self):
        """
        Get list of available actions from current state.
        Actions are (customer_id, vehicle_idx) tuples.
        """
        actions = []
        
        # Try adding customers to current vehicle
        for customer in self.unvisited:
            if self.can_add_customer(customer, self.current_vehicle):
                actions.append((customer, self.current_vehicle))
        
        # If current vehicle can't fit any more customers and there are unvisited ones,
        # allow moving to next vehicle
        if not actions and self.unvisited and self.current_vehicle < self.instance.max_vehicles - 1:
            # Add a special action to move to next vehicle
            actions.append(('NEXT_VEHICLE', self.current_vehicle + 1))
        
        return actions
    
    def apply_action(self, action):
        """Apply action and return new state."""
        new_state = self.clone()
        
        if action[0] == 'NEXT_VEHICLE':
            new_state.current_vehicle = action[1]
        else:
            customer, vehicle_idx = action
            new_state.routes[vehicle_idx].append(customer)
            new_state.unvisited.remove(customer)
        
        return new_state
    
    def calculate_total_distance(self):
        """Calculate total distance of all routes."""
        total_distance = 0
        depot = self.instance.depot
        
        for route in self.routes:
            if not route:
                continue
            
            # Depot to first customer
            total_distance += self.instance.get_distance(depot, route[0])
            
            # Between customers
            for i in range(len(route) - 1):
                total_distance += self.instance.get_distance(route[i], route[i + 1])
            
            # Last customer back to depot
            total_distance += self.instance.get_distance(route[-1], depot)
        
        return total_distance


# ============================================================================
# PART 3: MCTS NODE
# ============================================================================

class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state, parent=None, action=None):
        """
        Initialize MCTS node.
        
        Args:
            state: CVRPState object
            parent: Parent MCTSNode
            action: Action taken to reach this node from parent
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = state.get_available_actions()
        
    def is_fully_expanded(self):
        """Check if all actions from this node have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self):
        """Check if this is a terminal node."""
        return self.state.is_terminal()
    
    def best_child(self, exploration_weight=1.414):
        """
        Select best child using UCB1 formula.
        
        UCB1 = average_reward + exploration_weight * sqrt(ln(parent_visits) / child_visits)
        """
        choices_weights = []
        
        for child in self.children:
            if child.visits == 0:
                # Unvisited children get infinite weight
                weight = float('inf')
            else:
                # UCB1 formula
                exploitation = child.total_reward / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                weight = exploitation + exploration
            
            choices_weights.append((child, weight))
        
        return max(choices_weights, key=lambda x: x[1])[0]
    
    def expand(self):
        """Expand node by trying an untried action."""
        action = self.untried_actions.pop()
        next_state = self.state.apply_action(action)
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node
    
    def update(self, reward):
        """Update node statistics after simulation."""
        self.visits += 1
        self.total_reward += reward
    
    def get_best_action(self):
        """Get action leading to most visited child."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits).action


# ============================================================================
# PART 4: MCTS ALGORITHM
# ============================================================================

class MCTS_CVRP:
    """Monte Carlo Tree Search for CVRP."""
    
    def __init__(self, instance, exploration_weight=1.414, max_iterations=1000):
        """
        Initialize MCTS solver.
        
        Args:
            instance: CVRPInstance object
            exploration_weight: UCB1 exploration parameter
            max_iterations: Maximum number of MCTS iterations
        """
        self.instance = instance
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        self.root = None
        self.best_solution = None
        self.best_distance = float('inf')
        self.worst_distance = 0
        self.iteration_history = []
        self.all_rewards = []  # Track all rewards for normalization
        
    def search(self, verbose=True):
        """
        Execute MCTS search.
        
        Returns:
            best_state: Best CVRPState found
        """
        initial_state = CVRPState(self.instance)
        self.root = MCTSNode(initial_state)
        
        if verbose:
            print(f"Starting MCTS search with {self.max_iterations} iterations...")
            print(f"Problem: {self.instance.dimension} nodes, {self.instance.max_vehicles} vehicles")
            print(f"Optimal solution: 784")
            print("-" * 60)
        
        for iteration in range(self.max_iterations):
            # Dynamic exploration weight - decrease over time but not too much
            # This keeps exploration alive longer
            progress = iteration / self.max_iterations
            current_exploration = self.exploration_weight * (0.5 + 0.5 * (1 - progress**0.5))
            
            # MCTS four phases: Selection, Expansion, Simulation, Backpropagation
            node = self._select(self.root, current_exploration)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
            
            # Track progress
            if iteration % 100 == 0 and verbose:
                print(f"Iteration {iteration}: Best distance = {self.best_distance:.2f} (exploration={current_exploration:.3f})")
            
            self.iteration_history.append({
                'iteration': iteration,
                'best_distance': self.best_distance,
                'tree_size': self._count_nodes(self.root)
            })
        
        # Final improvement: Apply 2-opt to best solution one more time
        if self.best_solution:
            self.best_solution = self._apply_2opt_improvement(self.best_solution)
            self.best_distance = self.best_solution.calculate_total_distance()
        
        if verbose:
            print("-" * 60)
            print(f"Search complete!")
            print(f"Best distance found: {self.best_distance:.2f}")
            print(f"Gap from optimal: {((self.best_distance - 784) / 784 * 100):.2f}%")
        
        return self.best_solution
    
    def _select(self, node, exploration_weight=None):
        """
        Selection phase: Navigate tree using UCB1 until leaf or unexpanded node.
        """
        if exploration_weight is None:
            exploration_weight = self.exploration_weight
            
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child(exploration_weight)
        return node
    
    def _simulate(self, node):
        """
        Simulation phase: Complete the solution using a rollout policy.
        Returns negative distance (we want to minimize distance).
        """
        state = node.state.clone()
        
        # Complete the solution using nearest neighbor heuristic
        while not state.is_terminal():
            actions = state.get_available_actions()
            
            if not actions:
                # No valid actions, return poor reward
                return -10000
            
            # Choose action using greedy nearest neighbor
            action = self._greedy_action(state, actions)
            state = state.apply_action(action)
        
        # Apply 2-opt local search to improve route order
        state = self._apply_2opt_improvement(state)
        
        # Calculate total distance
        distance = state.calculate_total_distance()
        
        # Track worst distance for normalization
        if distance > self.worst_distance and distance < 10000:
            self.worst_distance = distance
        
        # Update best solution
        if distance < self.best_distance:
            self.best_distance = distance
            self.best_solution = state
        
        # Normalized reward: map distances to [0, 1] range
        # Better solutions get higher rewards
        if self.worst_distance > self.best_distance:
            normalized_reward = (self.worst_distance - distance) / (self.worst_distance - self.best_distance)
        else:
            normalized_reward = 0.5
        
        # Store for statistics
        self.all_rewards.append(normalized_reward)
        
        # Return normalized reward (0 to 1, higher is better)
        return normalized_reward
    
    def _greedy_action(self, state, actions):
        """
        Greedy heuristic: Choose customer closest to last customer in route.
        """
        # Filter out NEXT_VEHICLE actions for greedy selection
        customer_actions = [a for a in actions if a[0] != 'NEXT_VEHICLE']
        
        if not customer_actions:
            return actions[0]  # Return NEXT_VEHICLE if that's all we have
        
        depot = self.instance.depot
        current_route = state.routes[state.current_vehicle]
        
        if not current_route:
            # If route is empty, choose closest to depot
            last_node = depot
        else:
            last_node = current_route[-1]
        
        # Find closest customer
        best_action = None
        best_distance = float('inf')
        
        for action in customer_actions:
            customer = action[0]
            distance = self.instance.get_distance(last_node, customer)
            
            if distance < best_distance:
                best_distance = distance
                best_action = action
        
        return best_action if best_action else actions[0]
    
    def _apply_2opt_improvement(self, state):
        """
        Apply 2-opt local search to improve route order within each vehicle.
        This maintains the vehicle assignment but optimizes the visit sequence.
        """
        improved_state = state.clone()
        
        for vehicle_idx, route in enumerate(improved_state.routes):
            if len(route) < 2:
                continue
            
            improved = True
            while improved:
                improved = False
                
                # Try all possible 2-opt swaps
                for i in range(len(route) - 1):
                    for j in range(i + 2, len(route)):
                        # Calculate current distance for this segment
                        current_dist = self._calculate_segment_distance(
                            route, i, j, vehicle_idx, improved_state
                        )
                        
                        # Reverse the segment between i+1 and j
                        new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                        
                        # Calculate new distance
                        new_dist = self._calculate_segment_distance(
                            new_route, i, j, vehicle_idx, improved_state
                        )
                        
                        # If improvement found, apply it
                        if new_dist < current_dist:
                            improved_state.routes[vehicle_idx] = new_route
                            route = new_route
                            improved = True
                            break
                    
                    if improved:
                        break
        
        return improved_state
    
    def _calculate_segment_distance(self, route, i, j, vehicle_idx, state):
        """
        Calculate distance for a route segment affected by 2-opt.
        """
        depot = self.instance.depot
        distance = 0
        
        # Distance from previous node to route[i]
        if i == 0:
            distance += self.instance.get_distance(depot, route[i])
        else:
            distance += self.instance.get_distance(route[i-1], route[i])
        
        # Distance through the segment
        for k in range(i, j):
            distance += self.instance.get_distance(route[k], route[k+1])
        
        # Distance from route[j] to next node
        if j == len(route) - 1:
            distance += self.instance.get_distance(route[j], depot)
        else:
            distance += self.instance.get_distance(route[j], route[j+1])
        
        return distance
    
    def _backpropagate(self, node, reward):
        """
        Backpropagation phase: Update statistics for all nodes in path.
        Uses depth-aware weighting to give more credit to deeper decisions.
        """
        depth = 0
        while node is not None:
            # Depth-aware reward: deeper nodes get slightly higher weight
            # This encourages exploration of complete paths
            depth_bonus = 1.0 + (depth * 0.01)  # 1% bonus per depth level
            adjusted_reward = reward * depth_bonus
            
            node.update(adjusted_reward)
            node = node.parent
            depth += 1
    
    def _count_nodes(self, node):
        """Count total nodes in tree."""
        if node is None:
            return 0
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count


# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

class CVRPVisualizer:
    """Visualize CVRP solutions and MCTS progress."""
    
    def __init__(self, instance):
        """Initialize visualizer with CVRP instance."""
        self.instance = instance
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                       '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AAB7B8']
    
    def plot_solution(self, state, title="CVRP Solution"):
        """Plot a CVRP solution with routes."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot depot
        depot = self.instance.depot
        depot_x, depot_y = self.instance.coords[depot]
        ax.plot(depot_x, depot_y, 'r*', markersize=20, label='Depot', zorder=5)
        
        # Plot customers
        customers = self.instance.get_customers()
        for customer in customers:
            x, y = self.instance.coords[customer]
            demand = self.instance.demands[customer]
            ax.plot(x, y, 'ko', markersize=8, zorder=3)
            ax.text(x, y + 2, f'{customer}\n({demand})', ha='center', fontsize=8)
        
        # Plot routes
        total_distance = 0
        for idx, route in enumerate(state.routes):
            if not route:
                continue
            
            color = self.colors[idx % len(self.colors)]
            route_load = state.get_route_load(idx)
            
            # Depot to first customer
            x1, y1 = self.instance.coords[depot]
            x2, y2 = self.instance.coords[route[0]]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)
            total_distance += self.instance.get_distance(depot, route[0])
            
            # Between customers
            for i in range(len(route) - 1):
                x1, y1 = self.instance.coords[route[i]]
                x2, y2 = self.instance.coords[route[i + 1]]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)
                total_distance += self.instance.get_distance(route[i], route[i + 1])
            
            # Last customer back to depot
            x1, y1 = self.instance.coords[route[-1]]
            x2, y2 = self.instance.coords[depot]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7,
                   label=f'Vehicle {idx+1} (Load: {route_load}/{self.instance.capacity})')
            total_distance += self.instance.get_distance(route[-1], depot)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'{title}\nTotal Distance: {total_distance:.2f}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_convergence(self, mcts_solver):
        """Plot MCTS convergence over iterations."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        history = mcts_solver.iteration_history
        iterations = [h['iteration'] for h in history]
        best_distances = [h['best_distance'] for h in history]
        tree_sizes = [h['tree_size'] for h in history]
        
        # Plot best distance over iterations
        ax1.plot(iterations, best_distances, 'b-', linewidth=2)
        ax1.axhline(y=784, color='r', linestyle='--', label='Optimal (784)')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best Distance Found')
        ax1.set_title('MCTS Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot tree growth
        ax2.plot(iterations, tree_sizes, 'g-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Number of Nodes in Tree')
        ax2.set_title('MCTS Tree Growth')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_route_details(self, state):
        """Plot detailed information about each route."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        route_info = []
        for idx, route in enumerate(state.routes):
            if not route:
                continue
            
            load = state.get_route_load(idx)
            route_distance = self._calculate_route_distance(route)
            
            route_info.append({
                'Vehicle': f'V{idx+1}',
                'Customers': len(route),
                'Load': load,
                'Capacity': self.instance.capacity,
                'Utilization': f'{(load/self.instance.capacity*100):.1f}%',
                'Distance': f'{route_distance:.2f}'
            })
        
        # Create table
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [[info['Vehicle'], info['Customers'], f"{info['Load']}/{info['Capacity']}", 
                      info['Utilization'], info['Distance']] for info in route_info]
        
        table = ax.table(cellText=table_data,
                        colLabels=['Vehicle', 'Customers', 'Load', 'Utilization', 'Distance'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header
        for i in range(5):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows alternately
        for i in range(1, len(route_info) + 1):
            color = '#F0F0F0' if i % 2 == 0 else 'white'
            for j in range(5):
                table[(i, j)].set_facecolor(color)
        
        ax.set_title('Route Details', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    def _calculate_route_distance(self, route):
        """Calculate distance for a single route."""
        if not route:
            return 0
        
        depot = self.instance.depot
        distance = 0
        
        # Depot to first customer
        distance += self.instance.get_distance(depot, route[0])
        
        # Between customers
        for i in range(len(route) - 1):
            distance += self.instance.get_distance(route[i], route[i + 1])
        
        # Last customer back to depot
        distance += self.instance.get_distance(route[-1], depot)
        
        return distance


# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("MONTE CARLO TREE SEARCH FOR CVRP - TUTORIAL IMPLEMENTATION")
    print("=" * 70)
    print()
    
    # Load instance
    print("Loading CVRP instance...")
    instance = CVRPInstance('cvrp_instance.txt')
    print(f"Instance: {instance.name}")
    print(f"Customers: {instance.dimension - 1}")
    print(f"Vehicles: {instance.max_vehicles}")
    print(f"Vehicle Capacity: {instance.capacity}")
    print(f"Optimal Solution: 784")
    print()
    
    # Run MCTS
    print("Running MCTS...")
    mcts = MCTS_CVRP(instance, exploration_weight=5.0, max_iterations=10000)
    best_solution = mcts.search(verbose=True)
    print()
    
    # Create visualizer
    visualizer = CVRPVisualizer(instance)
    
    # Plot solution
    print("Generating visualizations...")
    fig1 = visualizer.plot_solution(best_solution, "Best CVRP Solution Found by MCTS")
    plt.savefig('cvrp_solution.png', dpi=150, bbox_inches='tight')
    print("✓ Solution plot saved as 'cvrp_solution.png'")
    
    # Plot convergence
    fig2 = visualizer.plot_convergence(mcts)
    plt.savefig('mcts_convergence.png', dpi=150, bbox_inches='tight')
    print("✓ Convergence plot saved as 'mcts_convergence.png'")
    
    # Plot route details
    fig3 = visualizer.plot_route_details(best_solution)
    plt.savefig('route_details.png', dpi=150, bbox_inches='tight')
    print("✓ Route details saved as 'route_details.png'")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best Distance: {best_solution.calculate_total_distance():.2f}")
    print(f"Optimal Distance: 784")
    print(f"Gap: {((best_solution.calculate_total_distance() - 784) / 784 * 100):.2f}%")
    print(f"Routes Used: {sum(1 for r in best_solution.routes if r)}")
    print()
    
    for idx, route in enumerate(best_solution.routes):
        if route:
            load = best_solution.get_route_load(idx)
            print(f"Vehicle {idx+1}: {route}")
            print(f"  Load: {load}/{instance.capacity} ({load/instance.capacity*100:.1f}% utilized)")
    
    plt.show()


if __name__ == "__main__":
    main()