## `dem_reg_map`
We replaced the nested μ×mode loops with a single broadcasted NumPy evaluation, so the entire grid is computed in one pass. μ-range handling is safer (finite/positive checks, geometric spacing), and the math runs in float64 with `np.errstate` to keep ratio/power operations stable. Temporary arrays are trimmed by leaning on broadcasting and direct reductions.  
- **Key wins:** loop → vectorized grid; safer μ bounds; fewer temporaries.

## `dem_inv_gsvd`
Anywhere the baseline did `A @ inv(B)`, we now solve linear systems (or use a pseudoinverse), which is both faster and better conditioned. Post-processing is also written in a vectorized style, scaling rows or columns directly rather than creating new diagonal matrices and multiplying them, which reduces both FLOPs and temporary allocations.
- **Key wins:** no explicit inverses; smaller/fewer matmuls; improved conditioning.

## `demmap_pos` (+ `dem_pix`)
demmap_pos the most aggressive changes because it sits on the hot path. First, we addressed parallel efficiency: if you use multiple processes, each process now limits internal BLAS threads to one so the system doesn’t oversubscribe and come to a halt while trying to create further threads. We keep chunking logic simple and deterministic to ensure large inputs don’t fall back to a slow serial remainder. Inside the pixel solver, we clean the inputs (finite checks, non-positive uncertainties) one time and then construct key matrices with broadcasted operations. The regularization filter is applied as a simple vector-scaling of the relevant factor matrix, avoiding repeated construction of big diagonal matrices and replacing multiple multiplications with cheaper row-scales plus a single matmul. The half-maximum width for elogt is computed from an interpolated profile using masks instead of Python control flow, which keeps the inner loop lighter and less costly. Overall, the inner loop reuses as much factored information as possible so that only the λ-dependent light pieces are recomputed each iteration. 
- **Key wins:** efficient inner loop (row-scale + one matmul); stable variance; efficient multiprocessing.

## General Changes
- **Vectorization first:** using numpy array-wide boolean masks/`np.where` instead of usual conditionals, rely on broadcasting, remove Python from tight loops.  
- **Numerical hygiene:** consistent float64, guarded parameter ranges, `solve/pinv` over `inv`, and clipping of round-off artefacts.  
- **Resource efficiency:** fewer temporaries and smaller matmuls reduce memory traffic; thread limits keep CPU utilization sane under multiprocessing.
