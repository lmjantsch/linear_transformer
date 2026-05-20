The objective of the framework is to linearise the model to allow for the forward propagation of an arbitrary part $\mathbf{x}_i$ where $\mathbf{x} = \sum_i \mathbf{x}_i$ or the backward propagation of any target vector $t$ such that the following symmetry condition holds:

$$f(\mathbf{x}_i) \bullet \mathbf{t} = \mathbf{x}_i \bullet f^{-1}(\mathbf{t})$$

To obtain such a condition, the following rules are applied.  

### Rule 1: Linear Transformation:
**Function:** $f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}$  
  
The bias term can be seen as its own component and is ignored in both forward and backward pass. Most modern LLMs further do not use biases anymore.  
- **Forward Decomposition:** $f(\mathbf{x}_i) = \mathbf{W}\mathbf{x}_i$
- **Backward Propagation:** $f^{-1}(\mathbf{t}) = \mathbf{W}^T \mathbf{t}$

### Rule 2: Zero-Preserving Nonlinearities ( $f(0) = 0$ )
**Function:** $f(\mathbf{x}) = \sigma(\mathbf{x})$  

For activation functions that naturally cross the origin, there is no constant baseline. Relevance is distributed strictly proportionally using an element-wise secant (the ratio of the total output to the total input).  
- **Forward Decomposition:** $f(\mathbf{x}_i) = \mathbf{x}_i \odot \frac{\sigma(\mathbf{x})}{\mathbf{x} + \epsilon \cdot \text{sign}(\mathbf{x})}$
- **Backward Propagation:** $f^{-1}(\mathbf{t}) = \mathbf{t} \odot \frac{\sigma(\mathbf{x})}{\mathbf{x} + \epsilon \cdot \text{sign}(\mathbf{x})}$

### Rule 3: Non-Zero Baseline Nonlinearities ( $f(0) \neq 0$ )
**Function:** $f(\mathbf{x}) = \sigma(\mathbf{x})$ (e.g., Softmax, GELU)  

For functions with a non-zero baseline, the secant rule causes numerical instability (division-by-zero) near $x=0$ . Deep Taylor Decomposition (DTD) is applied to formulate an effective, stabilized Jacobian matrix ( $\mathbf{J}_{DTD}$ ). This linearizes the function locally, allowing the $f(0)$ baseline to safely absorb relevance as a virtual bias.  
- **Forward Decomposition:** $f(\mathbf{x}_i) = \mathbf{J}_{DTD} \mathbf{x}_i$
- **Backward Propagation:** $f^{-1}(\mathbf{t}) = \mathbf{J}_{DTD}^T \mathbf{t}$

**Softmax Specifics:**
For $\mathbf{s} = \text{Softmax}(\mathbf{x})$ , the effective Jacobian is $\mathbf{J}_{DTD} = \mathbf{I} - \mathbf{1}\mathbf{s}^T$ . Applying this yields mathematically stable rules independent of the pre-softmax logits:  
- **Forward:** $f(\mathbf{x}_i) = \mathbf{x}_i - \mathbf{1}(\mathbf{s}^T \mathbf{x}_i)$
- **Backward:** $f^{-1}(\mathbf{t}) = \mathbf{t} - \mathbf{s}(\mathbf{1}^T \mathbf{t})$

### Rule 4: Bilinear Functions
**Function:** $f(\mathbf{x}, \mathbf{y}) = \mathbf{x} \odot \mathbf{y}$  

When two variable tensors multiply the vector must not be duplicated. The **Uniform Splitting Rule** dictates that the backward signal is distributed equally between the participating operands, granting half the relevance to the $\mathbf{x}$ pathway and half to the $\mathbf{y}$ pathway.  
- **Forward Decomposition** (w.r.t. $\mathbf{x}_i$ ): $f_x(\mathbf{x}_i) = \frac{1}{2} \mathbf{x}_i \odot \mathbf{y}$
- **Backward Propagation** (w.r.t. $\mathbf{x}$ ): $f_x^{-1}(\mathbf{t}) = \frac{1}{2} \mathbf{y} \odot \mathbf{t}$

## Components
### Norms

Normalization layers are evaluated by treating their statistical parameters (mean $\mu$ , standard deviation $\sigma$ , or RMS) as frozen scalars derived from the total forward pass. This reduces them to simple linear element-wise scaling (Rule 1), bypassing complex non-linear derivatives.  

**LayerNorm**
Function: $f(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sigma} \odot \gamma + \beta$  
*Reference State:* Effective weight $\mathbf{w} = \frac{\gamma}{\sigma}$ . (Per Rule 1, the effective bias $\beta - \frac{\gamma \mu}{\sigma}$ absorbs its own relevance and is ignored for the variable components).  
- **Forward Decomposition:** $f(\mathbf{x}_i) = \mathbf{w} \odot \mathbf{x}_i$
- **Backward Propagation:** $f^{-1}(\mathbf{t}) = \mathbf{w} \odot \mathbf{t}$

**RMSNorm**
Function: $f(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}} \odot \gamma$  
*Reference State:* Effective weight $\mathbf{w} = \frac{\gamma}{\text{RMS}}$ .  
- **Forward Decomposition:** $f(\mathbf{x}_i) = \mathbf{w} \odot \mathbf{x}_i$
- **Backward Propagation:** $f^{-1}(\mathbf{t}) = \mathbf{w} \odot \mathbf{t}$

### Attention
Attention relies on sequential applications of Rule 4 (Uniform Splitting) for the bilinear matrix multiplications and Rule 3 (Deep Taylor Decomposition) for the Softmax distribution.  

**MHSA**
*Reference States:* $\mathbf{Q}^* = \mathbf{X}\mathbf{W}_q$ , $\mathbf{K}^* = \mathbf{X}\mathbf{W}_k$ , $\mathbf{V}^* = \mathbf{X}\mathbf{W}_v$ , $\mathbf{A}^* = \text{Softmax}(\frac{\mathbf{Q}^* (\mathbf{K}^*)^T}{\sqrt{d}})$ .  
Because Rule 4 is applied twice (once at $\mathbf{A}\mathbf{V}$ , and once at $\mathbf{Q}\mathbf{K}^T$ ), the global relevance inherently splits as $\mu_v = \frac{1}{2}$ , $\mu_q = \frac{1}{4}$ , $\mu_k = \frac{1}{4}$ .  
- **Forward Decomposition:** $f(\mathbf{X}_i) = \big( f_q(\mathbf{X}_i) + f_k(\mathbf{X}_i) + f_v(\mathbf{X}_i) \big) \mathbf{W}_o$
    - $f_v(\mathbf{X}_i) = \frac{1}{2} \mathbf{A}^* (\mathbf{X}_i \mathbf{W}_v)$
    - $f_q(\mathbf{X}_i) = \frac{1}{4} \Big[ \frac{\mathbf{X}_i \mathbf{W}_q (\mathbf{K}^*)^T}{\sqrt{d}} - \mathbf{1} \Big( (\mathbf{A}^*)^T \frac{\mathbf{X}_i \mathbf{W}_q (\mathbf{K}^*)^T}{\sqrt{d}} \Big) \Big] \mathbf{V}^*$
    - $f_k(\mathbf{X}_i) = \frac{1}{4} \Big[ \frac{\mathbf{Q}^* (\mathbf{X}_i \mathbf{W}_k)^T}{\sqrt{d}} - \mathbf{1} \Big( (\mathbf{A}^*)^T \frac{\mathbf{Q}^* (\mathbf{X}_i \mathbf{W}_k)^T}{\sqrt{d}} \Big) \Big] \mathbf{V}^*$
- **Backward Propagation:** $f^{-1}(\mathbf{T}) = f_q^{-1}(\mathbf{T}) + f_k^{-1}(\mathbf{T}) + f_v^{-1}(\mathbf{T})$
    - $\tilde{\mathbf{T}} = \mathbf{T}\mathbf{W}_o^T$
    - $f_v^{-1}(\mathbf{T}) = \frac{1}{2} (\mathbf{A}^*)^T \tilde{\mathbf{T}} \mathbf{W}_v^T$
    - $\mathbf{T}_S = \frac{1}{2}\tilde{\mathbf{T}}(\mathbf{V}^*)^T - \mathbf{A}^* \odot \Big( \big( \frac{1}{2}\tilde{\mathbf{T}}(\mathbf{V}^*)^T \big) \mathbf{1}_N \Big)$ *(Softmax Rule 3 applied to Attention backward signal)*
    - $f_q^{-1}(\mathbf{T}) = \frac{1}{2} (\mathbf{T}_S / \sqrt{d}) \mathbf{K}^* \mathbf{W}_q^T$
    - $f_k^{-1}(\mathbf{T}) = \frac{1}{2} (\mathbf{T}_S^T / \sqrt{d}) \mathbf{Q}^* \mathbf{W}_k^T$

**MHSA + Rotary Embeddings**
Rotary Embeddings ( $\mathbf{R}_\Theta$ ) act as an intermediate linear transformation (Rule 1). Since rotation matrices are orthogonal, the backward pass requires a simple transpose multiplication.  
- **Forward Decomposition:** Replace $\mathbf{W}_q$ with $\mathbf{W}_q \mathbf{R}_\Theta$ and $\mathbf{W}_k$ with $\mathbf{W}_k \mathbf{R}_\Theta$ in the MHSA forward equations.
- **Backward Propagation:** The inverse rotation is applied strictly before the final weight projection.
    - $f_q^{-1}(\mathbf{T}) = \frac{1}{2} \big( (\mathbf{T}_S / \sqrt{d}) \mathbf{K}^* \big) \mathbf{R}_\Theta^T \mathbf{W}_q^T$
    - $f_k^{-1}(\mathbf{T}) = \frac{1}{2} \big( (\mathbf{T}_S^T / \sqrt{d}) \mathbf{Q}^* \big) \mathbf{R}_\Theta^T \mathbf{W}_k^T$

### Dense FFN
The feed-forward networks require tracing signals through zero-preserving activations (Rule 2) and, in the case of modern LLMs, internal bilinear splits (Rule 4).  

**MLP**
Function: $f(\mathbf{x}) = \sigma(\mathbf{x}\mathbf{W}_{up}) \mathbf{W}_{down}$  
*Reference State:* Secant ratio $\mathbf{c} = \frac{\sigma(\mathbf{x}\mathbf{W}_{up})}{\mathbf{x}\mathbf{W}_{up}}$ . (For GELU and Swish activation the secante cancels the $\mathbf{x}$ terms and leaves $\mathbf{c}_{GELU} = \Phi(\mathbf{x}\mathbf{W}_{up})$ where $\Phi$ is the Gaussian CDF and $\mathbf{c}_{SILU} = \sigma(\mathbf{x}\mathbf{W}_{up})$ where $\sigma$ is the sigmoid function).  
- **Forward Decomposition:** $f(\mathbf{x}_i) = \big( (\mathbf{x}_i \mathbf{W}_{up}) \odot \mathbf{c} \big) \mathbf{W}_{down}$
- **Backward Propagation:** $f^{-1}(\mathbf{t}) = \big( (\mathbf{t} \mathbf{W}_{down}^T) \odot \mathbf{c} \big) \mathbf{W}_{up}^T$

**GLU (Gated Linear Unit)**
Function: $f(\mathbf{x}) = \big( \sigma(\mathbf{x}\mathbf{W}_{gate}) \odot (\mathbf{x}\mathbf{W}_{up}) \big) \mathbf{W}_{down}$  
*Reference States:* Gate vector $\mathbf{G}^* = \sigma(\mathbf{x}\mathbf{W}_{gate})$ , Up vector $\mathbf{U}^* = \mathbf{x}\mathbf{W}_{up}$ , Gate Secant $\mathbf{c}_g = \frac{\sigma(\mathbf{x}\mathbf{W}_{gate})}{\mathbf{x}\mathbf{W}_{gate}}$  
- **Forward Decomposition:** $f(\mathbf{x}_i) = f_{gate}(\mathbf{x}_i) + f_{up}(\mathbf{x}_i)$
    - $f_{gate}(\mathbf{x}_i) = \frac{1}{2} \big( ((\mathbf{x}_i \mathbf{W}_{gate}) \odot \mathbf{c}_g) \odot \mathbf{U}^* \big) \mathbf{W}_{down}$
    - $f_{up}(\mathbf{x}_i) = \frac{1}{2} \big( \mathbf{G}^* \odot (\mathbf{x}_i \mathbf{W}_{up}) \big) \mathbf{W}_{down}$
- **Backward Propagation:** $f^{-1}(\mathbf{t}) = f_{gate}^{-1}(\mathbf{t}) + f_{up}^{-1}(\mathbf{t})$
    - $\tilde{\mathbf{t}} = \mathbf{t} \mathbf{W}_{down}^T$
    - $f_{gate}^{-1}(\mathbf{t}) = \frac{1}{2} \big( (\tilde{\mathbf{t}} \odot \mathbf{U}^*) \odot \mathbf{c}_g \big) \mathbf{W}_{gate}^T$
    - $f_{up}^{-1}(\mathbf{t}) = \frac{1}{2} \big( \tilde{\mathbf{t}} \odot \mathbf{G}^* \big) \mathbf{W}_{up}^T$

### MOE (Mixture of Experts)
A sparse MOE layer utilizes a router to calculate a probability distribution over $E$ experts. This is mathematically identical to an Attention mechanism but operating over the expert dimension instead of the sequence dimension.  

Function: $f(\mathbf{x}) = \sum_{e=1}^E p_e \cdot \text{Expert}_e(\mathbf{x})$  

*Reference States:* Router probabilities $\mathbf{P} = \text{Softmax}(\mathbf{x}\mathbf{W}_{route})$ , Expert outputs $\mathbf{E}_e = \text{Expert}_e(\mathbf{x})$ .  
- **Forward Decomposition:** $f(\mathbf{x}_i) = f_{router}(\mathbf{x}_i) + f_{experts}(\mathbf{x}_i)$
    - $f_{router}(\mathbf{x}_i) = \frac{1}{2} \sum_{e=1}^E \Big( (\mathbf{x}_i \mathbf{W}_{route})_e - \mathbf{P}^T (\mathbf{x}_i \mathbf{W}_{route}) \Big) \mathbf{E}_e$
    - $f_{experts}(\mathbf{x}_i) = \frac{1}{2} \sum_{e=1}^E p_e \cdot f_{expert\_e}(\mathbf{x}_i)$ *(Where f_{ expert_e} evaluates Rule 1 & 2 for the specific expert's MLP)*
- **Backward Propagation:** $f^{-1}(\mathbf{t}) = f_{router}^{-1}(\mathbf{t}) + \sum_{e=1}^E f_{expert\_e}^{-1}(\mathbf{t})$
    - $\mathbf{t}_P = \frac{1}{2} \sum_{e=1}^E (\mathbf{E}_e \odot \mathbf{t})$ *(Collapsing the target into the router dimension)*
    - $f_{router}^{-1}(\mathbf{t}) = \big( \mathbf{t}_P - \mathbf{P}(\mathbf{1}^T \mathbf{t}_P) \big) \mathbf{W}_{route}^T$ *(Applying Softmax Rule 3)*
    - $f_{expert\_e}^{-1}(\mathbf{t}) = \frac{1}{2} p_e \cdot \text{Expert}_e^{-1}(\mathbf{t})$ *(Passing the scaled target backward through the expert's MLP)*