# %% [markdown]
# # Eigenvalues & Eigenvectors — Hands-on Lab
#
# In the course, we solved this by hand for
#
# $$A = \begin{pmatrix}3 & 1\\1 & 3\end{pmatrix}$$
#
# and found:
#
# - det(A − λI) = (3−λ)² − 1 = 0  →  **λ = 4 or λ = 2**
# - λ = 4  →  eigenvector **v = [1, 1]**
# - λ = 2  →  eigenvector **v = [1, −1]**
#
# In this lab we verify that by hand result with code — first in **NumPy**
# (the plain math way), then in **PyTorch** (the way you'd actually see it
# inside an AI framework). Same numbers, two tools.

# %% [markdown]
# ## Part 1 — NumPy

# %%
import numpy as np

A = np.array([[3, 1],
              [1, 3]])
A

# %% [markdown]
# `np.linalg.eig` returns the eigenvalues and the matching eigenvectors
# (as columns of a matrix) in one call.

# %%
eigenvalues, eigenvectors = np.linalg.eig(A)

print("eigenvalues:\n", eigenvalues)
print("\neigenvectors (as columns):\n", eigenvectors)

# %% [markdown]
# Two things to notice before we move on:
#
# 1. The eigenvalues come out as `[4. 2.]` — same numbers we solved by hand.
# 2. The eigenvectors come out **normalized** (length 1), e.g. roughly
#    `[0.707, 0.707]` instead of `[1, 1]`. That's fine — an eigenvector is a
#    *direction*, not a specific size, so `[0.707, 0.707]` and `[1, 1]` point
#    the exact same way (NumPy just always returns the unit-length version).

# %% [markdown]
# ### Verify A·v = λ·v directly
#
# Let's check the definition itself, using our own hand-solved vectors
# (not normalized) so the numbers are easy to read.

# %%
def check_eigenpair(A, v, lam, label):
    v = np.array(v, dtype=float)
    Av = A @ v
    lam_v = lam * v
    print(f"{label}: A @ v = {Av}, λ·v = {lam_v}, match = {np.allclose(Av, lam_v)}")

check_eigenpair(A, [1, 1], 4, "λ=4, v=[1,1]")
check_eigenpair(A, [1, -1], 2, "λ=2, v=[1,-1]")

# %% [markdown]
# Both print `match = True` — confirming the hand calculation from the
# course. Try changing the numbers above (e.g. `[2, 2]` instead of `[1, 1]`)
# — it still matches, because any nonzero scalar multiple of an eigenvector
# is still an eigenvector.

# %% [markdown]
# ## Part 2 — PyTorch
#
# Same matrix, same question, but this time using the tensor library you'd
# actually use inside a neural network. The API is almost identical.

# %%
import torch

A_t = torch.tensor([[3.0, 1.0],
                     [1.0, 3.0]])
A_t

# %%
result = torch.linalg.eig(A_t)

print("eigenvalues:\n", result.eigenvalues)
print("\neigenvectors (as columns):\n", result.eigenvectors)

# %% [markdown]
# One PyTorch quirk: `torch.linalg.eig` always returns **complex** numbers
# (`4.+0.j`), even when the matrix is real and the eigenvalues have no
# imaginary part. That's because `eig` has to work for *any* square matrix,
# and some matrices (like a rotation) have genuinely complex eigenvalues —
# so PyTorch always uses the complex type to be safe. Since our eigenvalues
# are real, the imaginary part is just `0.j` and we can drop it with `.real`.

# %%
eigenvalues_t = result.eigenvalues.real
eigenvectors_t = result.eigenvectors.real

print("eigenvalues (real part):", eigenvalues_t)

# %% [markdown]
# ### Verify A·v = λ·v in PyTorch

# %%
def check_eigenpair_torch(A, v, lam, label):
    v = torch.tensor(v, dtype=torch.float32)
    Av = A @ v
    lam_v = lam * v
    print(f"{label}: A @ v = {Av}, λ·v = {lam_v}, match = {torch.allclose(Av, lam_v)}")

check_eigenpair_torch(A_t, [1.0, 1.0], 4.0, "λ=4, v=[1,1]")
check_eigenpair_torch(A_t, [1.0, -1.0], 2.0, "λ=2, v=[1,-1]")

# %% [markdown]
# ## Wrap-up
#
# - NumPy and PyTorch agree with each other and with the hand calculation:
#   λ = 4 with v = [1,1], and λ = 2 with v = [1,−1].
# - The only real difference is that `torch.linalg.eig` always returns
#   complex numbers, since PyTorch's `eig` has to handle any matrix,
#   including ones with genuinely complex eigenvalues (like rotations).
# - This is exactly what happens under the hood in real AI systems — e.g.
#   checking whether the eigenvalues of a weight matrix are above or below 1
#   to reason about exploding/vanishing gradients in an RNN.
#
# **Try it yourself:** change `A` to `[[1,1],[2,2]]` (the singular example
# from the course) and re-run Part 1. What happens to the eigenvalues?
