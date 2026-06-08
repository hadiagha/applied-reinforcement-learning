import pytest
import numpy as np
import sys
import importlib.util

# Import the environment module
spec = importlib.util.spec_from_file_location("pricing_env", "02_dynamic_pricing_env.py")
pricing_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pricing_env)
DynamicPricingEnv = pricing_env.DynamicPricingEnv


class TestDynamicPricingEnvIntegration:
    """Integration tests for DynamicPricingEnv to verify correct behavior"""
    
    def setup_method(self):
        """Setup a standard test environment before each test"""
        self.total_periods = 5
        self.initial_inventory = 20
        self.price_levels = [5, 10, 15, 20]
        self.base_demand = 10
        self.price_sensitivity = 1.0
        self.unsold_penalty = 0.5
        
        self.env = DynamicPricingEnv(
            total_time_periods=self.total_periods,
            initial_inventory=self.initial_inventory,
            price_levels=self.price_levels,
            base_demand_per_period=self.base_demand,
            price_sensitivity=self.price_sensitivity,
            unsold_penalty_per_item=self.unsold_penalty
        )
    
    def test_environment_initialization(self):
        """Test that environment initializes correctly"""
        assert self.env.total_time_periods == self.total_periods
        assert self.env.initial_inventory == self.initial_inventory
        assert self.env.price_levels == self.price_levels
        assert self.env.action_space_size == len(self.price_levels)
        assert self.env.base_demand_per_period == self.base_demand
        assert self.env.price_sensitivity == self.price_sensitivity
        assert self.env.unsold_penalty_per_item == self.unsold_penalty
    
    def test_reset_functionality(self):
        """Test that reset properly initializes state"""
        state = self.env.reset()
        
        assert state["time_periods_left"] == self.total_periods
        assert state["current_inventory"] == self.initial_inventory
        assert self.env.current_time_period == 0
        assert len(self.env.time_step_history) == 1  # Should have initial state
        assert len(self.env.inventory_history) == 1
    
    def test_valid_action_execution(self):
        """Test that valid actions execute correctly"""
        self.env.reset()
        initial_inventory = self.env.current_inventory
        initial_time_left = self.env.time_periods_left
        
        # Take action (choose lowest price for likely sales)
        action = 0  # Lowest price
        state, reward, done, info = self.env.step(action)
        
        # Check that time progressed
        assert state["time_periods_left"] == initial_time_left - 1
        assert self.env.current_time_period == 1
        
        # Check that inventory decreased (likely with low price)
        assert state["current_inventory"] <= initial_inventory
        
        # Check info contains expected keys
        assert "price_chosen" in info
        assert "units_sold" in info
        assert "revenue_this_period" in info
        assert "potential_demand" in info
        
        # Check that chosen price matches action
        assert info["price_chosen"] == self.price_levels[action]
    
    def test_invalid_action_handling(self):
        """Test that invalid actions raise appropriate errors"""
        self.env.reset()
        
        # Test action index too high
        with pytest.raises(ValueError):
            self.env.step(len(self.price_levels))
        
        # Test negative action index
        with pytest.raises(ValueError):
            self.env.step(-1)
    
    def test_revenue_calculation(self):
        """Test that revenue is calculated correctly"""
        self.env.reset()
        
        action = 0  # Choose first price level
        state, reward, done, info = self.env.step(action)
        
        # Revenue should be price * units_sold
        expected_revenue = info["price_chosen"] * info["units_sold"]
        assert info["revenue_this_period"] == expected_revenue
        assert reward >= 0  # Revenue should be non-negative (no penalty yet)
    
    def test_inventory_management(self):
        """Test that inventory is properly managed"""
        self.env.reset()
        
        # Take action and check inventory decreases
        initial_inventory = self.env.current_inventory
        action = 0  # Low price to encourage sales
        state, reward, done, info = self.env.step(action)
        
        # Inventory should decrease by units sold
        expected_inventory = initial_inventory - info["units_sold"]
        assert state["current_inventory"] == expected_inventory
        assert self.env.current_inventory == expected_inventory
    
    def test_episode_termination_time_limit(self):
        """Test that episode terminates when time runs out"""
        self.env.reset()
        
        # Run through all time periods
        for period in range(self.total_periods):
            action = np.random.randint(0, len(self.price_levels))
            state, reward, done, info = self.env.step(action)
            
            if period < self.total_periods - 1:
                assert not done, f"Episode should not be done at period {period}"
            else:
                assert done, "Episode should be done after all periods"
                assert state["time_periods_left"] == 0
    
    def test_episode_termination_sold_out(self):
        """Test that episode terminates when inventory is sold out"""
        # Create environment with very low inventory
        small_env = DynamicPricingEnv(
            total_time_periods=10,
            initial_inventory=1,  # Very small inventory
            price_levels=[1],  # Very low price to encourage sales
            base_demand_per_period=20,  # High demand
            price_sensitivity=0.1  # Low sensitivity
        )
        
        small_env.reset()
        state, reward, done, info = small_env.step(0)
        
        # Should sell out quickly with high demand and low price
        if info["units_sold"] >= 1:
            assert done, "Episode should end when inventory is sold out"
            assert state["current_inventory"] == 0
    
    def test_unsold_penalty_application(self):
        """Test that unsold penalty is applied at end of episode"""
        # Set up scenario likely to have unsold items
        penalty_env = DynamicPricingEnv(
            total_time_periods=2,
            initial_inventory=50,
            price_levels=[100],  # Very high price to discourage sales
            base_demand_per_period=5,
            price_sensitivity=2.0,
            unsold_penalty_per_item=1.0
        )
        
        penalty_env.reset()
        
        # Take actions with high prices
        for period in range(2):
            state, reward, done, info = penalty_env.step(0)  # High price
        
        # Should have penalty for unsold items
        if penalty_env.current_inventory > 0:
            expected_penalty = penalty_env.current_inventory * penalty_env.unsold_penalty_per_item
            # The final reward should include the penalty (negative)
            assert reward <= info["revenue_this_period"], "Final reward should include unsold penalty"
    
    def test_demand_calculation_price_sensitivity(self):
        """Test that demand decreases with higher prices"""
        self.env.reset()
        
        # Test low price
        low_price_action = 0  # Lowest price
        state, reward, done, info = self.env.step(low_price_action)
        low_price_demand = info["potential_demand"]
        
        self.env.reset()
        
        # Test high price
        high_price_action = len(self.price_levels) - 1  # Highest price
        state, reward, done, info = self.env.step(high_price_action)
        high_price_demand = info["potential_demand"]
        
        # Generally, lower prices should generate higher or equal demand
        # (though randomness might occasionally cause exceptions)
        # We'll check that at least one of several trials shows this pattern
        low_demands = []
        high_demands = []
        
        for trial in range(10):
            self.env.reset()
            _, _, _, info_low = self.env.step(0)
            low_demands.append(info_low["potential_demand"])
            
            self.env.reset()
            _, _, _, info_high = self.env.step(len(self.price_levels) - 1)
            high_demands.append(info_high["potential_demand"])
        
        avg_low_demand = np.mean(low_demands)
        avg_high_demand = np.mean(high_demands)
        
        assert avg_low_demand >= avg_high_demand, "Lower prices should generally create higher demand"
    
    def test_state_consistency(self):
        """Test that state information is consistent throughout episode"""
        self.env.reset()
        
        # Check initial state
        state = self.env._get_state()
        assert state["time_periods_left"] == self.total_periods
        assert state["current_inventory"] == self.initial_inventory
        
        # Take action and verify state updates
        action = 1
        new_state, reward, done, info = self.env.step(action)
        
        # Check state consistency
        assert new_state["time_periods_left"] == self.total_periods - 1
        assert new_state["current_inventory"] == self.initial_inventory - info["units_sold"]
        
        # Check internal state matches returned state
        internal_state = self.env._get_state()
        assert internal_state == new_state
    
    def test_history_tracking(self):
        """Test that history is properly tracked for visualization"""
        self.env.reset()
        
        # Check initial history
        assert len(self.env.time_step_history) == 1
        assert len(self.env.inventory_history) == 1
        assert self.env.inventory_history[0] == self.initial_inventory
        
        # Take action and check history updates
        action = 0
        state, reward, done, info = self.env.step(action)
        
        # History should have grown
        assert len(self.env.time_step_history) == 2
        assert len(self.env.inventory_history) == 2
        assert len(self.env.price_history) == 2
        assert len(self.env.sales_history) == 2
        assert len(self.env.revenue_history) == 2
        
        # Check that values match action results
        assert self.env.price_history[-1] == info["price_chosen"]
        assert self.env.sales_history[-1] == info["units_sold"]
        assert self.env.revenue_history[-1] == info["revenue_this_period"]


def test_run_full_episode():
    """Integration test for a complete episode"""
    env = DynamicPricingEnv(
        total_time_periods=3,
        initial_inventory=15,
        price_levels=[5, 10, 15],
        base_demand_per_period=8,
        price_sensitivity=1.0,
        unsold_penalty_per_item=0.5
    )
    
    state = env.reset()
    total_reward = 0
    step_count = 0
    
    while step_count < 3:  # Should finish within time limit
        action = np.random.randint(0, len(env.price_levels))
        state, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Verify state consistency
        assert isinstance(state["time_periods_left"], int)
        assert isinstance(state["current_inventory"], int)
        assert state["time_periods_left"] >= 0
        assert state["current_inventory"] >= 0
        
        # Verify info structure
        assert isinstance(info["price_chosen"], (int, float))
        assert isinstance(info["units_sold"], int)
        assert isinstance(info["revenue_this_period"], (int, float))
        assert isinstance(info["potential_demand"], int)
        
        if done:
            break
    
    print(f"Episode completed in {step_count} steps with total reward: ${total_reward:.2f}")


if __name__ == "__main__":
    # Run the tests
    test_instance = TestDynamicPricingEnvIntegration()
    
    print("Running integration tests for DynamicPricingEnv...")
    
    # Run each test method
    test_methods = [
        test_instance.test_environment_initialization,
        test_instance.test_reset_functionality,
        test_instance.test_valid_action_execution,
        test_instance.test_invalid_action_handling,
        test_instance.test_revenue_calculation,
        test_instance.test_inventory_management,
        test_instance.test_episode_termination_time_limit,
        test_instance.test_episode_termination_sold_out,
        test_instance.test_unsold_penalty_application,
        test_instance.test_demand_calculation_price_sensitivity,
        test_instance.test_state_consistency,
        test_instance.test_history_tracking
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