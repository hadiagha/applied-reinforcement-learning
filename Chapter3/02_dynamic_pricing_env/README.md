# 💰 Perishable product dynamic pricing environment

Imagine a perishable goods business where every pricing decision impacts your bottom line. You have limited time to sell your inventory before it expires, customer demand fluctuates with price changes, and unsold items become costly losses. This project demonstrates how intelligent pricing strategies emerge from simple business rules, showcasing the evolution from random pricing chaos to systematic revenue optimization.

## 📖 The Business Story

In today's competitive marketplace, pricing isn't just about covering costs—it's about maximizing revenue while managing inventory risk. Our dynamic pricing environment simulates the real-world challenge faced by businesses selling time-sensitive products: airlines with seats, hotels with rooms, retailers with perishable goods, or e-commerce platforms with limited-time offers.

This project brings that challenge to life through three interconnected components that demonstrate how artificial intelligence can transform business decision-making:

## 🏪 **The Core: Dynamic Pricing Environment** (`02_dynamic_pricing_env.py`)
*The complete business simulation where pricing intelligence emerges*

### What Makes This Environment Special

Our dynamic pricing environment is a fully-featured business simulation that captures the complexity of real-world pricing decisions:

**🌍 Rich Business Dynamics**
- **Time-constrained selling periods** with clear deadlines
- **Inventory management** with limited stock that decreases with sales
- **Price-sensitive demand** where customers respond to price changes
- **Stochastic market behavior** with realistic demand fluctuations
- **Real-time business visualization** showing performance metrics

**⚖️ Sophisticated Business Incentives**
The environment teaches optimal pricing through realistic business pressures:
- **Revenue generation**: Price × Quantity sold creates income
- **Time pressure**: Limited selling periods create urgency
- **Demand sensitivity**: Higher prices reduce customer demand
- **Inventory risk**: Unsold items incur penalty costs at episode end
- **Market dynamics**: Demand can increase as time pressure builds (urgency factor)

**🔄 Complete Business Cycle Management**
- **Episode structure**: Full selling seasons with clear start and end
- **State tracking**: Time remaining, inventory levels, pricing history
- **Performance metrics**: Revenue, sales volume, inventory turnover
- **Historical analysis**: Complete transaction history for strategy analysis

**🎨 Executive Dashboard**
- **Real-time analytics** with matplotlib integration
- **Multi-panel displays**: Inventory, pricing, sales, and revenue trends
- **Color-coded performance**: Visual indicators of business health
- **Time-series analysis**: Historical performance tracking
- **Interactive controls**: Adjustable visualization speed and detail

### Business Architecture

```python
class DynamicPricingEnv:
    """
    Complete business simulation featuring:
    - Time-constrained selling periods
    - Price-sensitive customer demand
    - Inventory management with penalties
    - Real-time business analytics
    - Configurable market parameters
    """
```

The environment operates on a realistic business cycle:
1. **Selling Phase**: Business has limited time to sell inventory
2. **Demand Response**: Customers react to pricing decisions  
3. **Revenue Generation**: Sales create income, unsold items create costs
4. **Performance Evaluation**: Success measured by total profit

### Key Business Features Deep Dive

**Demand Modeling**: The environment implements sophisticated demand curves where customer behavior responds realistically to price changes, including urgency effects as deadlines approach and stochastic market variations.

**Inventory Management**: Real-time inventory tracking with automatic deduction for sales, preventing overselling while maintaining accurate stock levels throughout the selling period.

**Penalty Structure**: Unsold inventory incurs realistic costs at episode end, creating authentic business pressure to balance revenue maximization with inventory clearance.

**Market Dynamics**: Advanced demand modeling includes urgency factors (customers more willing to buy as time runs out) and random market fluctuations that mirror real-world uncertainty.

## 🧪 **Validation Layer: Business Logic Tests** (`test_dynamic_pricing_integration.py`)
*Ensuring our business simulation operates flawlessly*

Before deploying pricing strategies in our environment, we must validate that every business rule works correctly. Our comprehensive test suite validates all aspects of the pricing environment:

### What Gets Tested

**🔍 Business Logic Validation**
- **Pricing mechanism**: Actions correctly map to price levels
- **Revenue calculation**: Sales properly generate income (price × quantity)
- **Inventory management**: Stock levels decrease accurately with sales
- **Time progression**: Business periods advance correctly

**📦 Market Behavior Verification**
- **Demand sensitivity**: Higher prices generate lower demand
- **Market response**: Customer behavior reacts appropriately to pricing
- **Sales execution**: Transactions process correctly within inventory limits
- **Episode termination**: Business cycles end at appropriate times

**🎯 Financial System Accuracy**
- **Revenue tracking**: Income calculations match sales reality
- **Penalty application**: Unsold inventory costs applied correctly
- **Performance metrics**: Business KPIs calculated accurately
- **State consistency**: Financial data remains internally consistent

**🔄 Edge Case Handling**
- **Inventory depletion**: Sold-out scenarios handled properly
- **Invalid pricing**: Out-of-range price selections rejected appropriately
- **Time expiration**: Episode ending triggers correct final calculations
- **Visualization data**: Historical tracking maintains data integrity

### Testing Philosophy

Our tests follow the principle of **business reality verification**: rather than testing internal code mechanics, we verify that the environment behaves like a real business from the decision-maker's perspective. This ensures our environment provides a reliable foundation for strategy development and comparison.

```python
def test_revenue_calculation():
    """Verify revenue equals price × quantity sold"""
    
def test_demand_price_sensitivity():
    """Confirm lower prices generate higher demand"""
    
def test_unsold_penalty_application():
    """Validate inventory costs applied at episode end"""
```

## 🎯 **Strategy Demonstration: Evolution Showcase** (`pricing_evolution_simple.ipynb`)
*Where business intelligence proves its worth*

With our environment validated and ready, the notebook demonstrates its power by showcasing the dramatic difference between random and intelligent pricing strategies.

### The Business Strategy Comparison

The notebook implements two fundamentally different approaches to the same business challenge:

**🎲 Random Agent: Business Chaos**
```python
class RandomAgent:
    def choose_action(self, state, price_levels):
        return np.random.randint(0, len(price_levels))  # Pure randomness
```
- Represents zero business intelligence - random price selection
- No consideration of market conditions or time pressure
- Inconsistent performance with high variance
- Provides baseline for measuring strategic improvement

**🧠 Heuristic Agent: Business Intelligence**
```python
class HeuristicAgent:
    def choose_action(self, state, price_levels):
        # Time-based pricing strategy
        # Early periods: premium pricing
        # Late periods: clearance pricing
```
- Implements strategic time-sensitive pricing
- Starts with premium prices when time is abundant
- Gradually reduces prices as deadline approaches
- Balances revenue maximization with inventory clearance

### What the Business Demonstration Reveals

**📊 Performance Metrics**
- **Revenue Generation**: Heuristic significantly outperforms random pricing
- **Inventory Management**: Strategic pricing reduces unsold stock
- **Consistency**: Heuristic delivers predictable business outcomes

**🎨 Strategic Evidence**
- **Pricing patterns**: Clear time-based strategy vs chaotic random decisions
- **Revenue curves**: Systematic improvement vs unpredictable results
- **Business intelligence**: Demonstrates value of strategic thinking

**🔬 Business Insights**
The notebook proves that our environment successfully:
- **Differentiates strategy quality**: Rewards intelligent vs random decisions
- **Provides consistent outcomes**: Reliable measurement of strategy effectiveness
- **Enables fair comparison**: Level playing field for strategy evaluation
- **Supports business learning**: Clear feedback for strategy development

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy matplotlib pytest jupyter
```

### The Complete Business Journey

**Step 1: Experience the Business Environment**
```bash
python 02_dynamic_pricing_env.py
```
Run the core environment to see random pricing decisions in action. This shows you the raw business environment with all its challenges and opportunities.

**Step 2: Verify Business Logic**
```bash
python test_dynamic_pricing_integration.py
```
Execute our comprehensive test suite. Green checkmarks confirm that every aspect of the business simulation operates correctly and reliably.

**Step 3: Witness Business Intelligence**
```bash
jupyter notebook pricing_evolution_simple.ipynb
```
Open the strategy demonstration notebook to see the dramatic transformation from random to intelligent pricing using the validated environment.

## 🎛️ Business Experimentation

The environment's flexible design enables extensive business scenario testing:

### Environment Configuration
```python
env = DynamicPricingEnv(
    total_time_periods=12,           # Selling season length
    initial_inventory=60,            # Starting stock levels
    price_levels=[8, 12, 16, 20],   # Available price points
    base_demand_per_period=15,       # Market size
    price_sensitivity=1.5,           # Price elasticity
    unsold_penalty_per_item=2.0      # Inventory carrying costs
)
```

### Business Research Questions

1. **How does market size affect pricing strategy effectiveness?**
   - Increase base demand and observe strategy adaptation
   - Measure performance gaps in different market conditions

2. **What's the optimal balance between premium and volume pricing?**
   - Adjust price levels and sensitivity parameters
   - Find the sweet spot for revenue maximization

3. **How does time pressure influence pricing decisions?**
   - Modify episode length and penalty structures
   - Discover optimal pricing acceleration patterns

4. **Can simple rules compete with complex algorithms?**
   - Compare heuristic strategies against more sophisticated approaches
   - Test the limits of rule-based business intelligence

## 🔬 The Business Science Behind the Environment

### Business Intelligence Foundations

Our environment implements core business concepts:

**Market Response**: `demand = f(price, time_pressure, randomness)`
- Captures realistic customer behavior patterns
- Models price elasticity and urgency effects

**Revenue Optimization**: `profit = revenue - inventory_costs`
- Simple but powerful profit maximization framework
- Balances income generation with cost management

**Time Value**: Episodes with clear deadlines
- Creates realistic business pressure
- Forces trade-offs between profit margin and inventory risk

**Performance Measurement**: Comprehensive business metrics
- Revenue, inventory turnover, profit margins
- Historical analysis for strategy evaluation



### Advanced Strategy Development
- Implement machine learning pricing algorithms
- Add competitor pricing and market dynamics
- Create multi-product portfolio optimization

### Market Enhancement
- Multiple customer segments with different price sensitivities
- Seasonal demand patterns and trend analysis
- Supply chain constraints and procurement costs

### Business Analytics
- Customer lifetime value calculations
- Market share and competitive analysis tools
- Real-time dashboard and KPI monitoring

## 📝 Quick Business Reference

| Component | Business Purpose | Entry Point |
|-----------|------------------|-------------|
| **Environment** | Core business simulation | `python 02_dynamic_pricing_env.py` |
| **Tests** | Business logic validation | `python test_dynamic_pricing_integration.py` |
| **Notebook** | Strategy demonstration | `jupyter notebook pricing_evolution_simple.ipynb` |

**Key Business Parameters:**
- `total_time_periods`: Length of selling season
- `initial_inventory`: Starting stock levels
- `price_levels`: Available pricing options
- `base_demand_per_period`: Market size
- `price_sensitivity`: Customer price elasticity
- `unsold_penalty_per_item`: Inventory carrying costs

## 🏆 Business Conclusion

You've just experienced one of the most fundamental phenomena in business intelligence: the emergence of strategic behavior from systematic rules. Our pricing environment's journey from random to intelligent mirrors the broader story of business AI—from gut instinct to data-driven optimization.

The next time you see dynamic pricing in action (airlines, hotels, ride-sharing), remember this simple demonstration. Behind every intelligent pricing decision lies the same basic principle: the right rules, applied systematically, can create remarkable business value.

Now go forth and optimize! Modify the parameters, experiment with new strategies, and discover your own insights into the fascinating world of business artificial intelligence.


## 📄 Copyright and Attribution

All rights reserved. This code and documentation are the intellectual property of **Manning Publications**  and **Hadi Aghazadeh**, author of "Reinforcement Learning for Business".

This dynamic pricing environment and associated demonstrations are created as educational materials to accompany the book's teachings on reinforcement learning principles and practical applications in business contexts.

**Book**: Reinforcement Learning for Business  
**Author**: Hadi Aghazadeh  
**Publisher**: Manning Publications  

For more information about the book and advanced reinforcement learning techniques, visit Manning Publications. 