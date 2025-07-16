# 🚛 3D Trailer Loading Environment

**Author**: Hadi Aghazadeh  
**Publisher**: Manning Publications  
**Book**: "Reinforcement Learning for Business"  
**Chapter**: 3 - Advanced Environment Design

## 📦 Overview

A **3D container packing environment** that simulates real-world trailer loading challenges. This environment teaches agents to intelligently pack items into multiple containers while maximizing space utilization and ensuring stable loading patterns.

Perfect for learning about **spatial reasoning**, **logistics optimization**, and **3D constraint satisfaction** in business applications.

### 🎯 Business Context

**Real-World Applications:**
- **Logistics**: Optimize trailer and container loading for shipping companies
- **Warehousing**: Maximize storage efficiency in distribution centers  
- **Transportation**: Improve load stability and space utilization
- **Supply Chain**: Reduce shipping costs through intelligent packing

## 🎮 How It Works

### **Environment Mechanics**
- **Multiple Containers**: Different trailer types with varying dimensions
- **3D Item Placement**: Items can be positioned anywhere in (x, y, z) coordinates
- **Orientation Control**: Each item can be rotated in 6 different ways
- **Physics Validation**: Prevents overlapping and ensures stable placement
- **Support Requirements**: Items need adequate support underneath for realistic loading

### **Key Features**
- **Smart Placement Scoring**: Rewards corner placement, wall contact, and stability
- **Visual 3D Rendering**: Professional trailer-style visualization with filled cuboids
- **Multiple Item Types**: Various box sizes with different colors and properties
- **Performance Metrics**: Track volume utilization, efficiency, and success rates

### **Agent Intelligence Levels**
- **Random Agent**: Chaotic placement with poor space utilization
- **Heuristic Agent**: Intelligent placement using support, corners, and gap minimization
- **Custom Agents**: Build your own strategies for optimal packing

## 📁 Files

### **`03_trailer_loading_env.py`**
Main environment file with complete 3D packing simulation, multiple container types, item definitions, and professional visualization.

### **`test_trailer_loading_integration.py`** 
Integration tests to validate environment mechanics, placement rules, and scoring systems.

### **`trailer_loading_evolution_simple.ipynb`**
Interactive notebook showing evolution from random to intelligent packing with side-by-side 3D visualizations.

## 🎛️ Easy Customization

### **Container Types**
Modify `TRAILER_SCHEMAS` to test different container sizes:
```python
TRAILER_SCHEMAS = [
    {'id': 'Container_A', 'dims': (8, 6, 5)},     # Standard
    {'id': 'Container_B', 'dims': (6, 6, 6)},     # Cube
    {'id': 'Container_C', 'dims': (10, 4, 4)},    # Long
]
```

### **Item Varieties**
Customize `ITEM_DEFINITIONS` for different packing scenarios:
```python
ITEM_DEFINITIONS = [
    {'id': 1, 'name': 'SmallBox', 'dims': (2, 2, 1), 'color': (1, 0, 0, 0.8)},
    {'id': 2, 'name': 'MedBox', 'dims': (3, 2, 2), 'color': (0, 1, 0, 0.8)},
    {'id': 3, 'name': 'LongBox', 'dims': (4, 1, 1), 'color': (0, 0, 1, 0.8)},
]
```

### **Loading Sequence**
Change `ITEM_SEQUENCE` to test different packing challenges:
```python
ITEM_SEQUENCE = ['SmallBox', 'MedBox', 'LongBox', 'SmallBox', 'MedBox']
```

## 🚀 Quick Start

### **1. Run the Environment**
```python
# Basic usage
python 03_trailer_loading_env.py
```

### **2. Test Different Strategies**
```python
# Open the evolution notebook
trailer_loading_evolution_simple.ipynb
```

### **3. Customize Your Scenario**
```python
# Edit the configurations at the top of the files:
CONTAINER_SCHEMAS = [...]  # Your container types
ITEM_DEFINITIONS = [...]   # Your item types  
ITEM_SEQUENCE = [...]      # Your packing sequence
```
 

## 📄 Copyright and Attribution

All rights reserved. This code and documentation are the intellectual property of **Manning Publications**  and **Hadi Aghazadeh**, author of "Reinforcement Learning for Business".

This dynamic pricing environment and associated demonstrations are created as educational materials to accompany the book's teachings on reinforcement learning principles and practical applications in business contexts.

**Book**: Reinforcement Learning for Business  
**Author**: Hadi Aghazadeh  
**Publisher**: Manning Publications 

For more information about the book and advanced reinforcement learning techniques, visit Manning Publications. 
