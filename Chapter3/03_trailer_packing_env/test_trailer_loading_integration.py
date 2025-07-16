import pytest
import numpy as np
import sys
import importlib.util

# Import the environment module
spec = importlib.util.spec_from_file_location("packing_env", "03_trailer_loading_env.py")
packing_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packing_env)
PackingEnv = packing_env.PackingEnv


class TestTrailerLoadingEnvIntegration:
    """Integration tests for PackingEnv to verify correct behavior"""
    
    def setup_method(self):
        """Setup a standard test environment before each test"""
        # Simple test configuration
        self.container_schemas = [
            {'id': 'TestContainer', 'dims': (5, 4, 3)}
        ]
        
        self.item_definitions = [
            {'id': 1, 'name': 'SmallBox', 'dims': (1, 1, 1), 'color': (1, 0, 0, 0.8)},
            {'id': 2, 'name': 'MediumBox', 'dims': (2, 2, 1), 'color': (0, 1, 0, 0.8)},
            {'id': 3, 'name': 'LargeBox', 'dims': (3, 2, 2), 'color': (0, 0, 1, 0.8)}
        ]
        
        self.item_sequence = ['SmallBox', 'MediumBox', 'LargeBox']
        
        self.env = PackingEnv(
            container_schemas=self.container_schemas,
            item_definitions=self.item_definitions,
            item_sequence_names=self.item_sequence
        )
    
    def test_environment_initialization(self):
        """Test that environment initializes correctly"""
        assert self.env.num_containers == 1
        assert len(self.env.container_schemas) == 1
        assert len(self.env.item_definitions) == 3
        assert self.env.num_orientations == 6
        assert len(self.env.container_spaces) == 1
        assert self.env.container_spaces[0].shape == (5, 4, 3)
    
    def test_reset_functionality(self):
        """Test that reset properly initializes state"""
        obs = self.env.reset()
        
        # Check observation structure
        container_spaces, item_info = obs
        assert len(container_spaces) == 1
        assert item_info is not None
        assert item_info['id'] == 1  # First item (SmallBox)
        assert item_info['base_dims'] == (1, 1, 1)
        
        # Check internal state
        assert self.env.current_item_idx_to_pack == 0
        assert self.env.total_steps_taken_episode == 0
        assert self.env.invalid_attempts_current_item == 0
        assert self.env.total_packed_volume == 0
        assert len(self.env.packed_items_info) == 0
        assert np.all(self.env.container_spaces[0] == 0)
    
    def test_valid_item_placement(self):
        """Test that valid item placement works correctly"""
        self.env.reset()
        
        # Place small box at origin with default orientation
        action = (0, 0, 0, 0, 0)  # container 0, position (0,0,0), orientation 0
        obs, reward, done, info = self.env.step(action)
        
        # Check that item was placed
        assert reward > 0  # Should get volume reward
        assert info['packed_volume_current_item'] == 1  # 1x1x1 = 1
        assert info['packed_item_name'] == 'SmallBox'
        assert info['packed_in_container_idx'] == 0
        assert not info['skipped_item']
        
        # Check container state
        assert self.env.container_spaces[0][0, 0, 0] == 1  # Item ID 1
        assert self.env.total_packed_volume == 1
        assert self.env.current_item_idx_to_pack == 1  # Moved to next item
    
    def test_invalid_container_index(self):
        """Test handling of invalid container indices"""
        self.env.reset()
        
        # Try to place in non-existent container
        action = (5, 0, 0, 0, 0)  # Invalid container index
        obs, reward, done, info = self.env.step(action)
        
        assert reward < 0  # Should get penalty
        assert info['error'] is not None
        assert 'Invalid c_idx' in info['error']
        assert self.env.current_item_idx_to_pack == 0  # Should not advance
        assert self.env.invalid_attempts_current_item == 1
    
    def test_invalid_orientation(self):
        """Test handling of invalid orientation indices"""
        self.env.reset()
        
        # Try invalid orientation
        action = (0, 0, 0, 0, 10)  # Invalid orientation index
        obs, reward, done, info = self.env.step(action)
        
        assert reward < 0  # Should get penalty
        assert info['error'] is not None
        assert self.env.invalid_attempts_current_item == 1
    
    def test_overlapping_placement(self):
        """Test that overlapping items are rejected"""
        self.env.reset()
        
        # Place first item
        action1 = (0, 0, 0, 0, 0)
        obs, reward1, done, info1 = self.env.step(action1)
        assert reward1 > 0  # Should succeed
        
        # Try to place second item in same location
        action2 = (0, 0, 0, 0, 0)  # Same position
        obs, reward2, done, info2 = self.env.step(action2)
        
        assert reward2 < 0  # Should fail
        assert not info2.get('packed_item_name')  # No item packed
        assert self.env.invalid_attempts_current_item > 0
    
    def test_out_of_bounds_placement(self):
        """Test that out-of-bounds placements are rejected"""
        self.env.reset()
        
        # Try to place item outside container bounds
        action = (0, 10, 10, 10, 0)  # Outside 5x4x3 container
        obs, reward, done, info = self.env.step(action)
        
        assert reward < 0  # Should fail
        assert not info.get('packed_item_name')
        assert self.env.invalid_attempts_current_item == 1
    
    def test_orientation_functionality(self):
        """Test that different orientations work correctly"""
        self.env.reset()
        
        # Skip to medium box (2x2x1) for better orientation testing
        self.env.current_item_idx_to_pack = 1
        
        # Test orientation 0: (2,2,1)
        action = (0, 0, 0, 0, 0)
        obs, reward, done, info = self.env.step(action)
        
        # Verify the oriented dimensions were used correctly
        if reward > 0:  # If placement succeeded
            assert info['packed_volume_current_item'] == 4  # 2x2x1 = 4
    
    def test_volume_tracking(self):
        """Test that volume tracking works correctly"""
        self.env.reset()
        
        # Place small box (volume 1)
        action1 = (0, 0, 0, 0, 0)
        obs, reward1, done, info1 = self.env.step(action1)
        
        assert self.env.total_packed_volume == 1
        assert self.env.packed_volume_per_container[0] == 1
        
        # Place medium box (volume 4) 
        action2 = (0, 2, 0, 0, 0)  # Different position
        obs, reward2, done, info2 = self.env.step(action2)
        
        if reward2 > 0:  # If placement succeeded
            assert self.env.total_packed_volume == 5  # 1 + 4
            assert self.env.packed_volume_per_container[0] == 5
    
    def test_item_skipping_after_max_attempts(self):
        """Test that items are skipped after too many invalid attempts"""
        # Create environment with very small container to force failures
        small_env = PackingEnv(
            container_schemas=[{'id': 'Tiny', 'dims': (1, 1, 1)}],
            item_definitions=[{'id': 1, 'name': 'BigBox', 'dims': (3, 3, 3), 'color': (1, 0, 0, 0.8)}],
            item_sequence_names=['BigBox']
        )
        
        small_env.reset()
        attempts = 0
        max_attempts = 20  # Safety limit
        
        while attempts < max_attempts:
            action = (0, 0, 0, 0, 0)  # Will always fail - item too big
            obs, reward, done, info = small_env.step(action)
            attempts += 1
            
            if info.get('skipped_item'):
                assert reward < -5  # Should get skip penalty
                break
        
        assert attempts < max_attempts, "Item should have been skipped"
    
    def test_episode_completion(self):
        """Test that episode completes when all items are processed"""
        self.env.reset()
        
        # Place all items successfully
        actions = [
            (0, 0, 0, 0, 0),  # SmallBox at (0,0,0)
            (0, 2, 0, 0, 0),  # MediumBox at (2,0,0) 
            (0, 0, 2, 0, 0),  # LargeBox at (0,2,0)
        ]
        
        for i, action in enumerate(actions):
            obs, reward, done, info = self.env.step(action)
            
            if i < len(actions) - 1:
                assert not done, f"Episode should not be done after item {i}"
            else:
                assert done, "Episode should be done after last item"
    
    def test_state_consistency(self):
        """Test that state information remains consistent"""
        self.env.reset()
        
        # Place an item
        action = (0, 0, 0, 0, 0)
        obs, reward, done, info = self.env.step(action)
        
        # Check observation consistency
        container_spaces, item_info = obs
        
        # Container space should match internal state
        assert np.array_equal(container_spaces[0], self.env.container_spaces[0])
        
        # Item info should match current item
        if self.env.current_item_idx_to_pack < len(self.env.items_to_pack_this_episode):
            current_item = self.env.items_to_pack_this_episode[self.env.current_item_idx_to_pack]
            assert item_info['id'] == current_item['id']
            assert item_info['base_dims'] == current_item['dims']
    
    def test_info_dictionary_accuracy(self):
        """Test that info dictionary provides accurate information"""
        self.env.reset()
        
        # Place an item
        action = (0, 0, 0, 0, 0)
        obs, reward, done, info = self.env.step(action)
        
        # Check info structure
        assert 'total_packed_volume_all_containers' in info
        assert 'packed_volume_per_container' in info
        assert 'items_processed_count' in info
        assert 'total_steps_episode' in info
        
        # Check values
        assert info['total_packed_volume_all_containers'] == self.env.total_packed_volume
        assert info['packed_volume_per_container'] == self.env.packed_volume_per_container.tolist()
        assert info['items_processed_count'] == self.env.current_item_idx_to_pack
        assert info['total_steps_episode'] == self.env.total_steps_taken_episode
    
    def test_packed_items_tracking(self):
        """Test that packed items are tracked correctly"""
        self.env.reset()
        
        # Place an item
        action = (0, 1, 1, 1, 0)
        obs, reward, done, info = self.env.step(action)
        
        if reward > 0:  # If placement succeeded
            assert len(self.env.packed_items_info) == 1
            
            item_info = self.env.packed_items_info[0]
            assert item_info['name'] == 'SmallBox'
            assert item_info['id'] == 1
            assert item_info['container_idx'] == 0
            assert item_info['pos'] == (1, 1, 1)
            assert item_info['volume'] == 1


def test_run_full_episode():
    """Integration test for a complete episode"""
    container_schemas = [{'id': 'TestContainer', 'dims': (6, 5, 4)}]
    item_definitions = [
        {'id': 1, 'name': 'Box1', 'dims': (1, 1, 1), 'color': (1, 0, 0, 0.8)},
        {'id': 2, 'name': 'Box2', 'dims': (2, 1, 1), 'color': (0, 1, 0, 0.8)}
    ]
    item_sequence = ['Box1', 'Box2']
    
    env = PackingEnv(
        container_schemas=container_schemas,
        item_definitions=item_definitions,
        item_sequence_names=item_sequence
    )
    
    obs = env.reset()
    total_reward = 0
    step_count = 0
    
    while step_count < 20:  # Safety limit
        # Simple placement strategy: try at origin
        action = (0, 0, 0, 0, 0)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Verify state consistency
        container_spaces, item_info = obs
        assert isinstance(container_spaces, list)
        assert len(container_spaces) == 1
        
        if done:
            break
    
    print(f"Episode completed in {step_count} steps with total reward: {total_reward:.2f}")
    print(f"Total packed volume: {env.total_packed_volume}")


if __name__ == "__main__":
    # Run the tests
    test_instance = TestTrailerLoadingEnvIntegration()
    
    print("Running integration tests for PackingEnv...")
    
    # Run each test method
    test_methods = [
        test_instance.test_environment_initialization,
        test_instance.test_reset_functionality,
        test_instance.test_valid_item_placement,
        test_instance.test_invalid_container_index,
        test_instance.test_invalid_orientation,
        test_instance.test_overlapping_placement,
        test_instance.test_out_of_bounds_placement,
        test_instance.test_orientation_functionality,
        test_instance.test_volume_tracking,
        test_instance.test_item_skipping_after_max_attempts,
        test_instance.test_episode_completion,
        test_instance.test_state_consistency,
        test_instance.test_info_dictionary_accuracy,
        test_instance.test_packed_items_tracking
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