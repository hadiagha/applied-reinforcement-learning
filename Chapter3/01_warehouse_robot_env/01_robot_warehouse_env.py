import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class OrderPickingEnv:
    """
    Enhanced Order Picking Robot Environment with Obstacles and Depot Return.

    The agent controls a robot to pick up items in a grid warehouse
    with obstacles, and then return to the starting depot.
    The goal is to complete the order efficiently.
    """

    def __init__(self, grid_size=(7, 7), depot_pos=(0, 0), 
                 item_locations=None, obstacle_locations=None,
                 max_steps_per_episode=75): # Increased max_steps
        """
        Initializes the warehouse environment.

        Args:
            grid_size (tuple): (rows, cols) of the grid.
            depot_pos (tuple): (row, col) for robot's starting/return depot.
            item_locations (list of tuples): List of (row, col) for each item.
            obstacle_locations (list of tuples): List of (row, col) for obstacles.
            max_steps_per_episode (int): Maximum steps allowed per episode.
        """
        self.grid_rows, self.grid_cols = grid_size
        self.depot_pos = np.array(depot_pos)

        # Obstacles
        self.obstacle_locations = [np.array(loc) for loc in obstacle_locations] if obstacle_locations else []

        # Item locations (ensure they are not on obstacles or depot initially)
        default_items = [
            np.array([1, 3]),
            np.array([max(0,min(self.grid_rows-1, self.grid_rows // 2 + 1)), max(0,min(self.grid_cols-1, self.grid_cols - 2))]),
            np.array([max(0,min(self.grid_rows-1, self.grid_rows - 2)), max(0,min(self.grid_cols-1, self.grid_cols // 2 -1))])
        ]
        self.item_locations = [np.array(loc) for loc in item_locations] if item_locations else default_items
        self.num_items = len(self.item_locations)

        # --- Validation for initial positions ---
        for obs_loc in self.obstacle_locations:
            if np.array_equal(self.depot_pos, obs_loc):
                raise ValueError(f"Depot position {self.depot_pos} cannot be on an obstacle {obs_loc}.")
            for item_loc in self.item_locations:
                if np.array_equal(item_loc, obs_loc):
                    raise ValueError(f"Item location {item_loc} cannot be on an obstacle {obs_loc}.")
        if self.num_items == 0:
            print("Warning: No item locations specified.")


        self.action_space_size = 4  # 0:Up, 1:Down, 2:Left, 3:Right
        self.max_steps = max_steps_per_episode

        # Internal state
        self.robot_pos = None
        self.items_picked_status = None
        self.all_items_collected_phase = False # Tracks if main picking phase is done
        self.current_step = 0
        
        # For rendering
        self.fig, self.ax = None, None
        self.robot_plot_object = None
        self.item_plot_objects = []
        self.obstacle_patches = []

        self.reset()

    def reset(self):
        self.robot_pos = np.array(self.depot_pos) # Start at depot
        self.items_picked_status = [False] * self.num_items
        self.all_items_collected_phase = False
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        return {
            "robot_pos": tuple(self.robot_pos),
            "items_picked": tuple(self.items_picked_status),
            "all_items_collected_phase": self.all_items_collected_phase 
        }

    def step(self, action_idx):
        if not (0 <= action_idx < self.action_space_size):
            raise ValueError(f"Invalid action {action_idx}. Must be 0, 1, 2, or 3.")

        self.current_step += 1
        
        move_deltas = [np.array([-1, 0]), np.array([1, 0]), 
                       np.array([0, -1]), np.array([0, 1])]
        delta = move_deltas[action_idx]
        potential_new_pos = self.robot_pos + delta

        movement_reward = -0.1 # Default cost for any action attempt
        
        # Check for obstacle collision
        is_obstacle_collision = any(np.array_equal(potential_new_pos, obs_loc) for obs_loc in self.obstacle_locations)

        if is_obstacle_collision:
            movement_reward = -0.7 # Higher penalty for hitting obstacle, robot does not move
        elif (0 <= potential_new_pos[0] < self.grid_rows) and \
             (0 <= potential_new_pos[1] < self.grid_cols):
            self.robot_pos = potential_new_pos
            # movement_reward stays -0.1
        else: # Bumped into wall (boundary)
            movement_reward = -0.5 # Robot does not move

        item_pickup_reward = 0.0
        if not self.all_items_collected_phase: # Can only pick items if not in return phase
            for i, item_loc in enumerate(self.item_locations):
                if not self.items_picked_status[i] and np.array_equal(self.robot_pos, item_loc):
                    self.items_picked_status[i] = True
                    item_pickup_reward = 15.0 # Increased bonus for picking an item
                    if all(self.items_picked_status):
                        self.all_items_collected_phase = True 
                        # Optional: small bonus here for finishing collection
                        # item_pickup_reward += 5.0 
                    break 
        
        done = False
        completion_reward = 0.0

        if self.all_items_collected_phase and np.array_equal(self.robot_pos, self.depot_pos):
            completion_reward = 100.0 # Large bonus for completing order AND returning to depot
            done = True
        
        if self.current_step >= self.max_steps and not done:
            done = True
            # Penalty if task not fully completed (all items picked AND returned to depot)
            if not (self.all_items_collected_phase and np.array_equal(self.robot_pos, self.depot_pos)):
                completion_reward = -20.0 

        total_reward = movement_reward + item_pickup_reward + completion_reward
        
        info = {
            "items_left_to_pick": self.num_items - sum(self.items_picked_status) if not self.all_items_collected_phase else 0,
            "in_return_to_depot_phase": self.all_items_collected_phase
        }
        return self._get_state(), total_reward, done, info

    def render(self, mode='human', pause_duration=0.1):
        if mode != 'human':
            return

        if self.fig is None: # First time setup
            self.fig, self.ax = plt.subplots(figsize=(7, 7)) # Slightly larger for clarity
            plt.ion()

            self.ax.set_xlim(-0.5, self.grid_cols - 0.5)
            self.ax.set_ylim(self.grid_rows - 0.5, -0.5)
            self.ax.set_xticks(np.arange(self.grid_cols))
            self.ax.set_yticks(np.arange(self.grid_rows))
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.grid(True, which='major', color='grey', linestyle='-', linewidth=0.7)
            self.ax.set_aspect('equal', adjustable='box')

            # Plot obstacles
            self.obstacle_patches = []
            for obs_loc in self.obstacle_locations:
                rect = patches.Rectangle((obs_loc[1]-0.5, obs_loc[0]-0.5), 1, 1, 
                                         linewidth=1, edgecolor='black', facecolor='dimgray', hatch='xx')
                self.ax.add_patch(rect)
                self.obstacle_patches.append(rect) # Keep ref if needed, though add_patch is enough

            # Plot depot
            self.ax.plot(self.depot_pos[1], self.depot_pos[0], marker='H', markersize=15, 
                         color='gold', markeredgecolor='black', label='Depot')

            # Plot initial item locations
            self.item_plot_objects = []
            for i, loc in enumerate(self.item_locations):
                item_marker, = self.ax.plot(loc[1], loc[0], marker='s', markersize=12, 
                                           color='deepskyblue', linestyle='None', label=f'Item {i}')
                self.item_plot_objects.append(item_marker)
            
            self.robot_plot_object, = self.ax.plot(self.robot_pos[1], self.robot_pos[0], marker='o', 
                                                  markersize=15, color='orangered', label='Robot')
            # self.ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.01, 1))


        self.robot_plot_object.set_data([self.robot_pos[1]], [self.robot_pos[0]])

        for i, item_marker_obj in enumerate(self.item_plot_objects):
            if self.items_picked_status[i]:
                item_marker_obj.set_marker('x')
                item_marker_obj.set_color('lightcoral')
                item_marker_obj.set_markersize(14)
            else:
                item_marker_obj.set_marker('s')
                item_marker_obj.set_color('deepskyblue')
                item_marker_obj.set_markersize(12)

        picked_count = sum(self.items_picked_status)
        phase_str = "Returning to Depot" if self.all_items_collected_phase else "Picking Items"
        self.ax.set_title(f"Step: {self.current_step}/{self.max_steps} | Items: {picked_count}/{self.num_items} | Phase: {phase_str}", fontsize=9)

        self.fig.canvas.draw_idle()
        plt.pause(pause_duration)

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None

# --- Main section to demonstrate the environment ---
if __name__ == '__main__':
    GRID_R, GRID_C = 8, 10 # Rectangular grid
    DEPOT = (GRID_R -1, 0) # Bottom-left corner as depot

    # Define items, ensuring they are not the depot
    custom_item_locs = [(1, 1), (1, GRID_C - 2), (GRID_R - 2, GRID_C // 2 + 1), (GRID_R // 2 -1, 2)]
    
    # Define obstacles, ensuring they are not items or depot
    custom_obstacle_locs = [
        (2,2), (2,3), (2,4), (2,5), (2,6), # A horizontal wall
        (GRID_R-3, GRID_C-3), (GRID_R-4, GRID_C-3), (GRID_R-5, GRID_C-3) # A vertical wall segment
    ]
    # Filter out any obstacles that might accidentally overlap with items/depot for this demo
    non_conflicting_obstacles = []
    for obs in custom_obstacle_locs:
        if np.array_equal(np.array(obs), np.array(DEPOT)): continue
        if any(np.array_equal(np.array(obs), np.array(item)) for item in custom_item_locs): continue
        non_conflicting_obstacles.append(obs)


    env = OrderPickingEnv(grid_size=(GRID_R, GRID_C), 
                          depot_pos=DEPOT, 
                          item_locations=custom_item_locs,
                          obstacle_locations=non_conflicting_obstacles,
                          max_steps_per_episode=100) # Allow more steps
    
    state = env.reset()
    print("Starting Random Agent Demonstration for Enhanced Order Picking Robot...")
    print(f"Warehouse Grid Size: {GRID_R}x{GRID_C}")
    print(f"Robot Depot: {env.depot_pos}")
    print(f"Target Item Locations: {env.item_locations}")
    print(f"Obstacle Locations: {env.obstacle_locations}")
    print(f"Initial State: Robot at {state['robot_pos']}, Items Picked: {state['items_picked']}, Return Phase: {state['all_items_collected_phase']}")
    
    env.render(pause_duration=1.0)

    total_reward_episode = 0.0
    done = False

    for step_num in range(1, env.max_steps + 1):
        action = np.random.randint(0, env.action_space_size)
        
        next_state, reward, done, info = env.step(action)
        total_reward_episode += reward
        
        # Simplified console output for cleaner demo run
        if step_num % 5 == 0 or done: # Print every 5 steps or if done
             print(f"Step {step_num}: Action: {action}, RobPos: {next_state['robot_pos']}, "
                   f"Picked: {sum(next_state['items_picked'])}/{env.num_items}, "
                   f"ReturnPhase: {next_state['all_items_collected_phase']}, Reward: {reward:.1f}, CumRew: {total_reward_episode:.1f}")

        env.render(pause_duration=0.1)
        state = next_state

        if done:
            print(f"\nEpisode finished after {step_num} steps.")
            if state['all_items_collected_phase'] and np.array_equal(np.array(state['robot_pos']), env.depot_pos):
                print("  SUCCESS: All items collected AND robot returned to depot!")
            elif state['all_items_collected_phase']:
                print("  PARTIAL SUCCESS: All items collected, but robot did NOT return to depot.")
            elif env.current_step >= env.max_steps:
                 print("  FAILURE: Max steps reached. Order not fully completed.")
            else: # Should be covered
                 print("  Order not fully completed for other reasons.")
            break
    
    if not done:
        print(f"\nEpisode finished after {env.max_steps} steps (Max steps reached via loop).")
        print("  FAILURE: Max steps reached. Order not fully completed.")

    print(f"\nFinal Robot Position: {state['robot_pos']}")
    print(f"Final Items Picked Status: {state['items_picked']}")
    print(f"Total Reward for Episode: {total_reward_episode:.2f}")

    print("\nClose the plot window to exit.")
    plt.show(block=True)
    env.close()