import pytest
import numpy as np
import sys
import importlib.util

# Import the environment module
spec = importlib.util.spec_from_file_location("robot_env", "01_robot_warehouse_env.py")
robot_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(robot_env)
OrderPickingEnv = robot_env.OrderPickingEnv


class TestOrderPickingEnvIntegration:
    """Integration tests for OrderPickingEnv to verify correct behavior"""
    
    def setup_method(self):
        """Setup a standard test environment before each test"""
        # Simple 5x5 grid for testing
        self.grid_size = (5, 5)
        self.depot_pos = (0, 0)
        self.item_locations = [(2, 2), (4, 4)]
        self.obstacle_locations = [(1, 1), (2, 1), (3, 3)]
        
        self.env = OrderPickingEnv(
            grid_size=self.grid_size,
            depot_pos=self.depot_pos,
            item_locations=self.item_locations,
            obstacle_locations=self.obstacle_locations,
            max_steps_per_episode=50
        )
    
    def test_obstacle_collision_reward(self):
        """Test that bumping into obstacles gives correct penalty and doesn't move robot"""
        self.env.reset()
        
        # Move robot to position (0,1) which is next to obstacle at (1,1)
        self.env.robot_pos = np.array([0, 1])
        initial_pos = self.env.robot_pos.copy()
        
        # Try to move down (action 1) which should hit obstacle at (1,1)
        state, reward, done, info = self.env.step(1)  # Down
        
        # Should get obstacle collision penalty and not move
        assert reward == -0.7, f"Expected obstacle collision reward -0.7, got {reward}"
        assert np.array_equal(self.env.robot_pos, initial_pos), "Robot should not move when hitting obstacle"
        assert state["robot_pos"] == tuple(initial_pos), "State should reflect robot hasn't moved"
    
    def test_wall_collision_reward(self):
        """Test that bumping into walls gives correct penalty and doesn't move robot"""
        self.env.reset()
        
        # Robot starts at (0,0), try to go up (out of bounds)
        initial_pos = self.env.robot_pos.copy()
        
        state, reward, done, info = self.env.step(0)  # Up (out of bounds)
        
        # Should get wall collision penalty and not move
        assert reward == -0.5, f"Expected wall collision reward -0.5, got {reward}"
        assert np.array_equal(self.env.robot_pos, initial_pos), "Robot should not move when hitting wall"
    
    def test_normal_movement_reward(self):
        """Test that normal movement gives correct penalty"""
        self.env.reset()
        
        # Move right from (0,0) to (0,1) - should be valid
        state, reward, done, info = self.env.step(3)  # Right
        
        assert reward == -0.1, f"Expected normal movement reward -0.1, got {reward}"
        assert self.env.robot_pos[0] == 0 and self.env.robot_pos[1] == 1, "Robot should move to (0,1)"
    
    def test_item_pickup_reward(self):
        """Test that picking up items gives correct reward and updates state"""
        self.env.reset()
        
        # Move robot to position adjacent to first item location (2,2)
        # Item is at (2,2), so position robot at (2,1) and move right
        self.env.robot_pos = np.array([2, 1])
        initial_items_picked = sum(self.env.items_picked_status)
        
        # Move right to the item location
        state, reward, done, info = self.env.step(3)  # Right
        
        # Should get item pickup reward plus movement reward
        expected_reward = 15.0 + (-0.1)  # item pickup + movement cost
        assert reward == expected_reward, f"Expected reward {expected_reward}, got {reward}"
        assert sum(self.env.items_picked_status) == initial_items_picked + 1, "Should have picked up one item"
        assert state["items_picked"][0] == True, "First item should be marked as picked"
    
    def test_multiple_item_pickup_sequence(self):
        """Test picking up multiple items in sequence"""
        self.env.reset()
        
        # Move to first item (2,2) - position robot adjacent and move into item
        self.env.robot_pos = np.array([2, 1])
        state, reward, done, info = self.env.step(3)  # Right to pick up first item
        
        assert sum(state["items_picked"]) == 1, "Should have picked up first item"
        assert not state["all_items_collected_phase"], "Should not be in return phase yet"
        
        # Move to second item (4,4) - position robot adjacent and move into item
        self.env.robot_pos = np.array([4, 3])
        state, reward, done, info = self.env.step(3)  # Right to pick up second item
        
        assert sum(state["items_picked"]) == 2, "Should have picked up both items"
        assert state["all_items_collected_phase"], "Should now be in return phase"
    
    def test_completion_reward(self):
        """Test that returning to depot after collecting all items gives completion reward"""
        self.env.reset()
        
        # Manually set all items as picked and enter return phase
        self.env.items_picked_status = [True, True]
        self.env.all_items_collected_phase = True
        
        # Move robot to depot
        self.env.robot_pos = np.array(self.depot_pos)
        
        state, reward, done, info = self.env.step(0)  # Any action
        
        expected_reward = 100.0 + (-0.5)  # completion + wall hit (going up from depot)
        assert reward == expected_reward, f"Expected completion reward {expected_reward}, got {reward}"
        assert done == True, "Episode should be complete"
    
    def test_timeout_penalty(self):
        """Test that exceeding max steps without completion gives penalty"""
        # Create environment with very few steps
        short_env = OrderPickingEnv(
            grid_size=(3, 3),
            depot_pos=(0, 0),
            item_locations=[(2, 2)],
            obstacle_locations=[],
            max_steps_per_episode=2
        )
        
        short_env.reset()
        
        # Take actions to exceed max steps without completing
        state, reward, done, info = short_env.step(0)  # Step 1
        assert not done, "Should not be done after step 1"
        
        state, reward, done, info = short_env.step(0)  # Step 2 (max steps reached)
        
        assert done == True, "Should be done after reaching max steps"
        assert reward <= -20.0, f"Should get timeout penalty, got {reward}"
    
    def test_obstacle_maze_navigation(self):
        """Test navigating around obstacles to reach items"""
        self.env.reset()
        
        # Try to navigate from (0,0) to item at (2,2) while avoiding obstacles
        # Path: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)
        
        moves = [3, 3, 1, 1]  # Right, Right, Down, Down
        positions = [(0,1), (0,2), (1,2), (2,2)]
        
        for i, action in enumerate(moves):
            state, reward, done, info = self.env.step(action)
            expected_pos = positions[i]
            assert state["robot_pos"] == expected_pos, f"Step {i+1}: Expected {expected_pos}, got {state['robot_pos']}"
            
            # Last move should pick up item
            if i == len(moves) - 1:
                assert reward > 10, f"Should get item pickup reward, got {reward}"
                assert sum(state["items_picked"]) == 1, "Should have picked up item"
    
    def test_cannot_pick_items_in_return_phase(self):
        """Test that items cannot be picked up during return phase"""
        self.env.reset()
        
        # Manually set return phase but place robot on unpicked item
        self.env.all_items_collected_phase = True
        self.env.items_picked_status = [False, False]  # Items not actually picked
        self.env.robot_pos = np.array([2, 2])  # On first item location
        
        state, reward, done, info = self.env.step(0)  # Any action
        
        # Should not pick up item since in return phase
        assert state["items_picked"][0] == False, "Should not pick up item in return phase"
        assert reward <= 0, "Should not get item pickup reward in return phase"
    
    def test_state_consistency(self):
        """Test that state information is consistent throughout episode"""
        self.env.reset()
        
        # Check initial state
        state = self.env._get_state()
        assert state["robot_pos"] == tuple(self.depot_pos), "Should start at depot"
        assert all(not picked for picked in state["items_picked"]), "No items should be picked initially"
        assert not state["all_items_collected_phase"], "Should not be in return phase initially"
        
        # Move and check state updates
        new_state, reward, done, info = self.env.step(3)  # Move right
        assert new_state["robot_pos"] == (0, 1), "State should reflect new position"
    
    def test_info_dictionary_accuracy(self):
        """Test that info dictionary provides accurate information"""
        self.env.reset()
        
        # Initial info
        _, _, _, info = self.env.step(0)  # Any action
        assert info["items_left_to_pick"] == 2, "Should have 2 items left to pick"
        assert not info["in_return_to_depot_phase"], "Should not be in return phase"
        
        # Pick up one item - position adjacent and move into item
        self.env.robot_pos = np.array([2, 1])
        _, _, _, info = self.env.step(3)  # Right to pick up item
        assert info["items_left_to_pick"] == 1, "Should have 1 item left to pick"
        
        # Pick up second item - position adjacent and move into item
        self.env.robot_pos = np.array([4, 3])
        _, _, _, info = self.env.step(3)  # Right to pick up item
        assert info["items_left_to_pick"] == 0, "Should have 0 items left to pick"
        assert info["in_return_to_depot_phase"], "Should be in return phase"


def test_run_full_episode():
    """Integration test for a complete episode"""
    env = OrderPickingEnv(
        grid_size=(4, 4),
        depot_pos=(0, 0),
        item_locations=[(3, 3)],
        obstacle_locations=[(1, 1)],
        max_steps_per_episode=20
    )
    
    state = env.reset()
    total_reward = 0
    steps = 0
    
    while steps < 20:
        action = np.random.randint(0, 4)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Verify state consistency
        assert isinstance(state["robot_pos"], tuple), "Robot position should be tuple"
        assert isinstance(state["items_picked"], tuple), "Items picked should be tuple"
        assert isinstance(state["all_items_collected_phase"], bool), "Return phase should be boolean"
        
        if done:
            break
    
    print(f"Episode completed in {steps} steps with total reward: {total_reward}")


if __name__ == "__main__":
    # Run the tests
    test_instance = TestOrderPickingEnvIntegration()
    
    print("Running integration tests for OrderPickingEnv...")
    
    # Run each test method
    test_methods = [
        test_instance.test_obstacle_collision_reward,
        test_instance.test_wall_collision_reward,
        test_instance.test_normal_movement_reward,
        test_instance.test_item_pickup_reward,
        test_instance.test_multiple_item_pickup_sequence,
        test_instance.test_completion_reward,
        test_instance.test_timeout_penalty,
        test_instance.test_obstacle_maze_navigation,
        test_instance.test_cannot_pick_items_in_return_phase,
        test_instance.test_state_consistency,
        test_instance.test_info_dictionary_accuracy
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_instance.setup_method()
            test_method()
            print(f"✓ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_method.__name__}: {e}")
            failed += 1
    
    # Run the full episode test
    try:
        test_run_full_episode()
        print("✓ test_run_full_episode")
        passed += 1
    except Exception as e:
        print(f"✗ test_run_full_episode: {e}")
        failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. Please check the implementation.") 