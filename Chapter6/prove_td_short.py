import numpy as np
import matplotlib.pyplot as plt

# Business scenario: Restaurant table scheduling
# - Goal: Maximize revenue by deciding when to seat customers
# - Each hour: choose to seat a party now or wait for a larger party
# - Small party: $50, Large party: $120 (but only 30% chance each hour)
# - Restaurant closes after 4 hours

class RestaurantEnv:
    def __init__(self):
        self.hours_left = 4
        self.large_party_prob = 0.3
        self.small_party_revenue = 50
        self.large_party_revenue = 120
    
    def reset(self):
        self.hours_left = 4
        return self.hours_left
    
    def step(self, action):
        """
        action: 0 = wait for large party, 1 = seat small party now
        returns: next_state, reward, done
        """
        if action == 1:  # Seat small party immediately
            reward = self.small_party_revenue
            return 0, reward, True  # Restaurant filled, episode ends
        
        # Wait for large party
        self.hours_left -= 1
        if self.hours_left == 0:  # Closing time
            return 0, 0, True  # No revenue, restaurant closes
        
        # Check if large party arrives
        if np.random.rand() < self.large_party_prob:
            reward = self.large_party_revenue
            return 0, reward, True  # Seat large party, episode ends
        else:
            reward = 0  # No party this hour
            return self.hours_left, reward, False  # Continue waiting

def td_learning(episodes=1000, alpha=0.1, gamma=0.9):
    """Learn optimal scheduling policy using TD(0)"""
    env = RestaurantEnv()
    
    # Value function: V[hours_left] = expected revenue from this state
    V = np.zeros(5)  # States: 0, 1, 2, 3, 4 hours left
    errors = []
    
    for episode in range(episodes):
        state = env.reset()  # Start with 4 hours left
        
        while True:
            # Simple policy: wait if we have time, otherwise seat small party
            if state <= 1:
                action = 1  # Seat small party (desperate)
            else:
                action = 0  # Wait for large party
            
            next_state, reward, done = env.step(action)
            
            # TD Update: V(s) ← V(s) + α[r + γV(s') - V(s)]
            if done:
                target = reward  # No future states
            else:
                target = reward + gamma * V[next_state]
            
            td_error = target - V[state]
            V[state] += alpha * td_error
            
            # Track learning progress (error from optimal)
            total_error = sum(abs(v - optimal_values()[i]) for i, v in enumerate(V))
            errors.append(total_error)
            
            if done:
                break
            state = next_state
    
    return V, errors

def optimal_values():
    """Calculate true optimal values analytically for comparison"""
    # This is what TD learning should converge to
    gamma = 0.9 #Discounted Factor
    p = 0.3  # probability of large party
    large_party_revenue = 120
    
    # Working backwards from closing time:
    V = np.zeros(5)
    V[1] = max(50, p * large_party_revenue)  # 1 hour left: seat small ($50) or gamble ($36)
    V[2] = max(50, p * large_party_revenue + (1-p) * gamma * V[1])  # 2 hours left
    V[3] = max(50, p * large_party_revenue + (1-p) * gamma * V[2])  # 3 hours left  
    V[4] = max(50, p * large_party_revenue + (1-p) * gamma * V[3])  # 4 hours left
    
    return V

# Run TD learning simulation
np.random.seed(42)
learned_values, learning_curve = td_learning()
true_values = optimal_values()

# Show results
print("Restaurant Scheduling: Expected Revenue by Hours Remaining")
print("-" * 55)
print("Hours Left | TD Learned | True Optimal | Difference")
print("-" * 55)
for i in range(5):
    print(f"    {i}      |   ${learned_values[i]:6.2f}   |   ${true_values[i]:6.2f}    |   ${abs(learned_values[i] - true_values[i]):5.2f}")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(learning_curve[:2000], label='Total Value Error')  # First 2000 updates
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Perfect Learning')
plt.xlabel('TD Updates')
plt.ylabel('Total Absolute Error ($)')
plt.title('TD Learning Discovers Optimal Restaurant Scheduling Policy')
plt.legend()
plt.grid(True, alpha=0.1)
plt.savefig('td_convergence.png', dpi=300)
plt.show()

print(f"\nKey insight: With 4 hours left, expected revenue is ${learned_values[4]:.2f}")
print(f"This means waiting for large parties is worth ${learned_values[4] - 50:.2f} more than seating immediately!")