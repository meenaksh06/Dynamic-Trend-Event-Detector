import nbformat as nbf
import os

def update_interpretability_nb(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
    
    # 1. Add Mathematical Primer at the top
    math_md = """# Section 0: Mathematical Foundations 📐

Before interpreting the results, we must acknowledge the first-principles math that governs these models.

### 1. Linear Algebra (Attention)
The temporal attention weights are calculated via dot-product similarity in a projected subspace:
$$\\\\text{Attention}(Q, K) = \\\\text{softmax}\\\\left(\\\\frac{QK^T}{\\\\sqrt{d_k}}\\\\right)$$

### 2. Calculus (Gradients)
The **Saliency Maps** in Section 4 are derived from the **Multivariate Chain Rule**:
$$\\\\delta_{token} = \\\\left\\\\| \\\\frac{\\\\partial \\\\text{Output}}{\\\\partial \\\\text{Embedding}_{token}} \\\\right\\\\|_2$$

### 3. Optimization Topology
Our models navigate a high-dimensional **Loss Landscape**. The geometry (curvature) of this surface determines both convergence speed and generalization capability.
"""
    new_math_cell = nbf.v4.new_markdown_cell(math_md)
    nb.cells.insert(0, new_math_cell)
    
    # 2. Add Loss Landscape Visualizer at the end (before summary)
    viz_md = "## 5. Loss Landscape Visualization\nWe simulate a high-dimensional loss surface to visualize the difference between **Sharp** and **Flat** minima. Flat minima (wider basins) are mathematically associated with better generalization to new news trends."
    viz_code = """from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_landscape():
    # Simulate a loss surface using a mixture of Gaussians
    x = np.linspace(-2, 2, 80)
    y = np.linspace(-2, 2, 80)
    X, Y = np.meshgrid(x, y)
    
    # Mathematical representation of a non-convex landscape
    Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
    Z = -Z # Flip to show basins
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='terrain', edgecolor='none', alpha=0.9)
    
    # Annotate Sharp vs Flat
    ax.text(-0.5, -0.5, -0.5, "Sharp Minima (Overfitting)", color='red', weight='bold')
    ax.text(1.0, 1.0, -0.2, "Flat Minima (Generalizing)", color='blue', weight='bold')
    
    ax.set_title("Simulated Loss Landscape Geometry")
    ax.set_xlabel("Parameter Theta_1")
    ax.set_ylabel("Parameter Theta_2")
    ax.set_zlabel("Loss J(Theta)")
    plt.show()

plot_loss_landscape()"""
    
    new_viz_md = nbf.v4.new_markdown_cell(viz_md)
    new_viz_code = nbf.v4.new_code_cell(viz_code)
    
    # Find index of Section 5 or append
    idx = len(nb.cells) - 1
    for i, cell in enumerate(nb.cells):
        if "## 5. Summary" in cell.source:
            idx = i
            break
            
    nb.cells.insert(idx, new_viz_md)
    nb.cells.insert(idx + 1, new_viz_code)
    
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Updated {path} with mathematical sections.")

path = 'notebook-Phase-2/Model_Interpretability.ipynb'
if os.path.exists(path):
    update_interpretability_nb(path)
