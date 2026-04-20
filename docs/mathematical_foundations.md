# Mathematical Foundations of the Dynamic Trend & Event Detector

This document connects the implementation details of our multi-model pipeline to first-principles mathematics in **Linear Algebra**, **Multivariate Calculus**, and **Numerical Optimization**.

---

## 1. Linear Algebra: Semantic Projections & Attention

The core of our **BERTopic** and **Attention-LSTM** models relies on the geometry of high-dimensional vector spaces.

### 1.1 The Dot-Product as Semantic Proximity
In our Attention mechanism, we treat topic embeddings as vectors in $\mathbb{R}^n$. The "importance" of a features is determined by the **Scalar Projection** of a Query ($Q$) onto a Key ($K$):

$$\text{Score} = \frac{Q \cdot K}{\|Q\| \|K\|} = \cos(\theta)$$

In the **Scaled Dot-Product Attention** used by our transformers:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The scaling factor $\frac{1}{\sqrt{d_k}}$ is a first-principles correction: as dimensionality $d$ increases, the variance of the dot product grows, pushing the softmax into regions with extremely small gradients.

### 1.2 Manifold Learning (UMAP)
BERTopic uses **UMAP** for dimensionality reduction. Mathematically, this assumes the data is uniformly distributed on a local Riemannian manifold. UMAP uses **Fuzzy Simplicial Sets** to approximate this manifold structure in lower dimensions while preserving the topological features of the original news headlines.

---

## 2. Calculus: Gradient Flow in LSTMs

To forecast trends, our LSTM model must learn patterns across time. This is achieved via **Backpropagation Through Time (BPTT)**.

### 2.1 The Chain Rule & Vanishing Gradients
In a standard RNN, the gradient of the loss $L$ with respect to the weights $W$ involves a product of $t$ terms:
$$\frac{\partial L_t}{\partial W} \propto \prod_{k=1}^t \frac{\partial h_k}{\partial h_{k-1}}$$

If the eigenvalues of $\frac{\partial h_k}{\partial h_{k-1}}$ are $< 1$, the gradient vanishes exponentially. 

### 2.2 The LSTM Solution: Forget Gate Calculus
The LSTM cell state $C_t$ has a linear update rule:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Taking the partial derivative:
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t + (\text{small terms from } f_t, i_t, \tilde{C}_t \text{ dependence})$$

By setting the **forget gate** $f_t \approx 1$, the model creates a "gradient highway" where $\frac{\partial C_t}{\partial C_{t-1}} \approx 1$, allowing the gradient to flow backward across many months without decaying.

---

## 3. Optimization Geometry: The Loss Landscape

Deep Learning models are trained by navigating a non-convex **Loss Landscape** $J(\theta)$ in high-dimensional space.

### 3.1 Local Topology: The Hessian Matrix
The local curvature of the loss function is defined by the **Hessian Matrix** $H = \nabla^2 J(\theta)$:
- If $\text{eig}(H) > 0$, the model is in a **basin** (local minimum).
- If $\text{eig}(H)$ has mixed signs, it is at a **saddle point**.

### 3.2 Generalization Theory: Flat vs. Sharp Minima
Interpretability studies (and our SHAP analysis) often correlate with the "sharpness" of the minima:
- **Sharp Minima**: Large eigenvalues in $H$. Small perturbations in $\theta$ (data noise) cause large jumps in $J(\theta)$. High sensitivity.
- **Flat Minima**: Small eigenvalues in $H$. The model is robust to noise and generalizes better to unseen news trends.

### 3.3 Adam Optimizer: Adaptive Pre-conditioning
Standard SGD navigates the landscape with a constant step. The **Adam** optimizer used in our project adapts the learning rate $\eta$ for each parameter $j$:
$$\theta_{j, t+1} = \theta_{j, t} - \frac{\eta}{\sqrt{\hat{v}_{j, t}} + \epsilon} \hat{m}_{j, t}$$

Mathematically, $\frac{1}{\sqrt{\hat{v}_t}}$ acts as a diagonal approximation to the inverse of the Hessian matrix, effectively "flattening" the surface for more efficient descent.

---
> [!IMPORTANT]
> All interpretability techniques (SHAP, Saliency) are approximations of these underlying partial derivatives. Understanding the math ensures we stay grounded in the model's actual mechanics.
