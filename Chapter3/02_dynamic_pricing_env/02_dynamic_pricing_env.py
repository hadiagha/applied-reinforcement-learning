import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class DynamicPricingEnv:
    """
    Simplified Dynamic Pricing Environment for a Perishable Product.

    The agent needs to set prices over a fixed number of time periods
    to maximize revenue for a product with limited initial inventory.
    Demand is sensitive to price and may change as time runs out.
    """

    def __init__(self, total_time_periods=10, initial_inventory=50,
                 price_levels=None, base_demand_per_period=10,
                 price_sensitivity=2, unsold_penalty_per_item=0.5):
        """
        Initializes the environment.

        Args:
            total_time_periods (int): Total number of periods to sell.
            initial_inventory (int): Starting number of items.
            price_levels (list): List of discrete prices the agent can choose.
                                 Example: [5, 8, 10, 12]
            base_demand_per_period (int): Base demand if price was zero (conceptual).
            price_sensitivity (float): How much demand drops per unit increase in price.
            unsold_penalty_per_item (float): Cost for each item not sold by the end.
        """
        self.total_time_periods = total_time_periods
        self.initial_inventory = initial_inventory
        self.price_levels = price_levels if price_levels is not None else [5, 8, 10, 12]
        self.num_price_levels = len(self.price_levels)

        # Demand model parameters
        self.base_demand_per_period = base_demand_per_period
        self.price_sensitivity = price_sensitivity
        self.unsold_penalty_per_item = unsold_penalty_per_item

        # For RL typically, action space is 0 to N-1
        self.action_space_size = self.num_price_levels

        # Visualization attributes
        self.fig = None # Figure object
        self.axs = {}   # Dictionary to hold axes for subplots
        self._initialize_plot_history()

        self.reset()

    def _initialize_plot_history(self):
        """Initializes lists to store data for plotting."""
        self.time_step_history = []
        self.inventory_history = []
        self.price_history = []
        self.sales_history = []
        self.revenue_history = []
        self.cumulative_revenue_history = []

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
            dict: The initial state.
        """
        self.current_time_period = 0 # Tracks how many periods have passed
        self.time_periods_left = self.total_time_periods
        self.current_inventory = self.initial_inventory

        self._initialize_plot_history()
        # Initialize history with t=0 state
        self._update_history(price_chosen_for_period=None, units_sold_this_period=0, revenue_this_period=0)

        return self._get_state()

    def _get_state(self):
        """
        Returns the current state of the environment.
        """
        return {
            "time_periods_left": self.time_periods_left,
            "current_inventory": self.current_inventory,
        }

    def _calculate_demand(self, price_chosen):
        """
        Calculates the potential demand for the product at a given price.
        Demand is stochastic and can increase as time runs out (urgency).
        """
        base_price_effect = max(0, self.base_demand_per_period - self.price_sensitivity * price_chosen)
        urgency_factor = 1.0 + (self.total_time_periods - self.time_periods_left) / (2.0 * self.total_time_periods)
        noise = np.random.uniform(-0.1, 0.1) * base_price_effect
        potential_demand = int(round(max(0, base_price_effect * urgency_factor + noise)))
        return potential_demand

    def step(self, action_idx):
        """
        Executes one time step within the environment.

        Args:
            action_idx (int): Index corresponding to the chosen price level.

        Returns:
            tuple: (next_state, reward, done, info)
        """
        if not (0 <= action_idx < self.num_price_levels):
            raise ValueError(f"Invalid action. Action must be an index between 0 and {self.num_price_levels - 1}.")

        if self.time_periods_left <= 0:
            return self._get_state(), 0, True, {"message": "Episode already ended."}

        price_chosen = self.price_levels[action_idx]
        units_sold_this_period = 0
        potential_demand = 0 # Initialize potential_demand

        if self.current_inventory > 0:
            potential_demand = self._calculate_demand(price_chosen)
            units_sold_this_period = min(potential_demand, self.current_inventory)
        
        self.current_inventory -= units_sold_this_period
        revenue_this_period = price_chosen * units_sold_this_period
        reward = revenue_this_period
        
        self.time_periods_left -= 1
        self.current_time_period += 1
        
        done = self.time_periods_left <= 0 or self.current_inventory <= 0
        
        final_message = ""
        if done and self.time_periods_left <= 0: # End of all periods
            penalty_for_unsold = self.current_inventory * self.unsold_penalty_per_item
            reward -= penalty_for_unsold
            final_message = f"End of selling window. Unsold items: {self.current_inventory}. Penalty: {penalty_for_unsold:.2f}"
        elif done and self.current_inventory <= 0:
            final_message = "All items sold out."

        info = {
            "price_chosen": price_chosen, 
            "units_sold": units_sold_this_period,
            "revenue_this_period": revenue_this_period,
            "potential_demand": potential_demand, # Include actual potential demand
            "message": final_message
        }
        self._update_history(price_chosen, units_sold_this_period, revenue_this_period)
        return self._get_state(), reward, done, info

    def _update_history(self, price_chosen_for_period, units_sold_this_period, revenue_this_period):
        """Appends current step's data to history lists for plotting."""
        self.time_step_history.append(self.current_time_period)
        self.inventory_history.append(self.current_inventory)
        
        if price_chosen_for_period is not None: # This means it's an action step (not initial reset)
            self.price_history.append(price_chosen_for_period)
            self.sales_history.append(units_sold_this_period)
            self.revenue_history.append(revenue_this_period)
            current_cumulative_revenue = (self.cumulative_revenue_history[-1] if self.cumulative_revenue_history else 0) + revenue_this_period
            self.cumulative_revenue_history.append(current_cumulative_revenue)
        elif not self.price_history: # Handling the very first call from reset
             self.price_history.append(np.nan) # Use NaN for placeholder at t=0 for action-related plots
             self.sales_history.append(np.nan)
             self.revenue_history.append(np.nan)
             self.cumulative_revenue_history.append(0) # Cumulative revenue starts at 0

    def render(self, mode='human', pause_duration=0.5):
        """
        Visualizes the environment state with an improved layout.
        """
        if mode != 'human':
            return

        if self.fig is None: # First-time setup for the plot
            self.fig = plt.figure(figsize=(12, 7.5), constrained_layout=True)
            gs = gridspec.GridSpec(2, 2, figure=self.fig) 
            self.axs['inventory'] = self.fig.add_subplot(gs[0, 0])
            self.axs['price'] = self.fig.add_subplot(gs[0, 1])
            self.axs['sales'] = self.fig.add_subplot(gs[1, 0])
            self.axs['revenue'] = self.fig.add_subplot(gs[1, 1])
            plt.ion()

        for key in ['inventory', 'price', 'sales', 'revenue']:
            if key in self.axs:
                self.axs[key].clear()

        state = self._get_state()
        title_text = (
            f"Dynamic Pricing Simulation\n"
            f"Period: {self.current_time_period}/{self.total_time_periods} | "
            f"Time Left: {state['time_periods_left']} | "
            f"Inventory: {state['current_inventory']}"
        )
        current_total_revenue = 0.0
        if self.cumulative_revenue_history:
            current_total_revenue = self.cumulative_revenue_history[-1]
        title_text += f" | Total Revenue: ${current_total_revenue:.2f}"
        self.fig.suptitle(title_text, fontsize=14)

        plot_time_steps_actions = self.time_step_history[1:] if len(self.time_step_history) > 1 else []
        
        # Inventory Plot
        self.axs['inventory'].plot(self.time_step_history, self.inventory_history, marker='o', linestyle='-', color='b')
        self.axs['inventory'].set_title('Inventory Level Over Time')
        self.axs['inventory'].set_xlabel('Time Period')
        self.axs['inventory'].set_ylabel('Units in Stock')
        self.axs['inventory'].set_ylim(0, self.initial_inventory + max(5, self.initial_inventory * 0.05))

        # Price Chosen Plot
        # Use self.price_history directly, which should have NaNs for t=0 if handled correctly
        valid_price_indices = [i for i, p in enumerate(self.price_history) if not np.isnan(p)]
        valid_prices = [self.price_history[i] for i in valid_price_indices]
        valid_time_for_prices = [self.time_step_history[i] for i in valid_price_indices]

        if valid_prices:
             self.axs['price'].step(valid_time_for_prices, valid_prices, where='post', marker='s', color='r')
        self.axs['price'].set_title('Price Chosen Over Time')
        self.axs['price'].set_xlabel('Time Period')
        self.axs['price'].set_ylabel('Price ($)')
        if self.price_levels:
             self.axs['price'].set_ylim(min(self.price_levels) -1 , max(self.price_levels) + 1)
             self.axs['price'].set_yticks(self.price_levels)

        # Units Sold Plot
        valid_sales_indices = [i for i, s in enumerate(self.sales_history) if not np.isnan(s)]
        valid_sales = [self.sales_history[i] for i in valid_sales_indices]
        valid_time_for_sales = [self.time_step_history[i] for i in valid_sales_indices]
        if valid_sales:
            self.axs['sales'].bar(valid_time_for_sales, valid_sales, color='g', width=0.8)
        self.axs['sales'].set_title('Units Sold Per Period')
        self.axs['sales'].set_xlabel('Time Period')
        self.axs['sales'].set_ylabel('Units Sold')
        max_sales_val = max(1, max(valid_sales)) if valid_sales else 1
        self.axs['sales'].set_ylim(0, max_sales_val * 1.1)

        # Revenue Per Period Plot
        valid_revenue_indices = [i for i, r_val in enumerate(self.revenue_history) if not np.isnan(r_val)]
        valid_revenues = [self.revenue_history[i] for i in valid_revenue_indices]
        valid_time_for_revenues = [self.time_step_history[i] for i in valid_revenue_indices]

        if valid_revenues:
            self.axs['revenue'].bar(valid_time_for_revenues, valid_revenues, color='purple', width=0.8)
        self.axs['revenue'].set_title('Revenue Per Period')
        self.axs['revenue'].set_xlabel('Time Period')
        self.axs['revenue'].set_ylabel('Revenue ($)')
        max_revenue_val = max(1, max(valid_revenues)) if valid_revenues else 1
        self.axs['revenue'].set_ylim(0, max_revenue_val * 1.1)

        for ax_key in ['inventory', 'price', 'sales', 'revenue']:
            self.axs[ax_key].set_xlim(-0.5, self.total_time_periods + 0.5)
            tick_step = 1 if self.total_time_periods <= 10 else max(1, self.total_time_periods // 10)
            self.axs[ax_key].set_xticks(np.arange(0, self.total_time_periods + 1, step=tick_step))
            self.axs[ax_key].grid(True)

        plt.draw()
        plt.pause(pause_duration)

    def close(self):
        """Closes the visualization window."""
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.axs = {}

# --- Main section to demonstrate the environment ---
if __name__ == '__main__':
    TOTAL_PERIODS = 15
    INITIAL_STOCK = 100
    PRICE_OPTIONS = [8, 10, 12, 15, 18]
    BASE_DEMAND = 20
    PRICE_SENSITIVITY_FACTOR = 1.5
    UNSOLD_PENALTY = 1.0

    print("--- Dynamic Pricing RL Environment Demonstration ---")
    print(f"Total Periods: {TOTAL_PERIODS}, Initial Inventory: {INITIAL_STOCK}")
    print(f"Price Options: {PRICE_OPTIONS}\n")

    env = DynamicPricingEnv(
        total_time_periods=TOTAL_PERIODS,
        initial_inventory=INITIAL_STOCK,
        price_levels=PRICE_OPTIONS,
        base_demand_per_period=BASE_DEMAND,
        price_sensitivity=PRICE_SENSITIVITY_FACTOR,
        unsold_penalty_per_item=UNSOLD_PENALTY
    )

    state = env.reset()
    env.render(pause_duration=1.5)

    total_reward_episode = 0
    done = False

    for period_num in range(1, TOTAL_PERIODS + 1): # Loop for actual decision periods
        if done: # Check if done at the start of the loop
            break

        action = np.random.randint(0, env.action_space_size)
        chosen_price_for_print = env.price_levels[action]

        print(f"\n--- Period {env.current_time_period + 1} (Action for this period) ---") # current_time_period is 0-indexed for history
        print(f"State: Inv={state['current_inventory']}, Time Left={state['time_periods_left']}")
        print(f"Random Agent Action: Choose Price Level {action} (${chosen_price_for_print})")

        next_state, reward, done, info = env.step(action)
        total_reward_episode += reward

        print(f"  > Demand Model: Potential Demand was {info['potential_demand']}")
        print(f"  > Result: Sold {info['units_sold']} items at ${info['price_chosen']}.")
        print(f"  > Revenue this period: ${info['revenue_this_period']:.2f}. Reward this step: {reward:.2f}")
        print(f"  > Next State: Inv={next_state['current_inventory']}, Time Left={next_state['time_periods_left']}")
        if info.get("message"):
            print(f"  > Info: {info['message']}")

        state = next_state
        env.render(pause_duration=0.7)

        if done and period_num == TOTAL_PERIODS : # Ensure loop finishes if done due to other reasons
             print("\n--- Episode Finished (due to conditions before all periods elapsed or at end) ---")


    if not done: # If loop finished due to exhausting periods but not 'done' flag
        print("\n--- Episode Finished (all periods elapsed) ---")


    print(f"\nFinal Inventory: {env.current_inventory}")
    print(f"Total Cumulative Reward (Revenue - Penalties): ${total_reward_episode:.2f}")
    if env.cumulative_revenue_history:
        print(f"Total Gross Revenue (from sales only): ${env.cumulative_revenue_history[-1]:.2f}")

    print("\nClose the plot window to exit.")
    plt.ioff()
    plt.show()
    env.close()
