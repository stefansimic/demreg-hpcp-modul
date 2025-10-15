# Report CPU Vectorization

We set a clear goal: speed up the existing pipeline with CPU vectorization while keeping all public interfaces the same. First, we fixed a baseline using the benchmark script so we could compare speed and outputs after every change.

Our first change was in `dem_reg_map.py`. We replaced nested loops for the μ search with NumPy broadcasts. This computes the whole discrepancy curve in one pass. The tests showed the function stayed correct and became faster.

Next, we tried to optimize `dem_inv_gsvd.py`. Here we met shape and orientation edge cases (filters vs temperatures). Small edits led to dimension errors. To avoid risk, we reverted this file to the stable version and decided to optimize around it instead of inside it.

We then focused on `demmap_pos.py`. We kept the GSVD call exactly as in the original code and targeted the hot spots: whitening and building `kdag`. We vectorized the whitening so it uses array operations, and we replaced the per-filter loop that created the diagonal filter with a vectorized build. At first, this caused shape mismatches when multiplying by `W`. We fixed that by creating the exact `(nt × nf)` block that the original code implied: place the `(nf × nf)` core at the top and zero-pad the rest. This matched the original math for both `nf < nt` and `nf ≥ nt`, removed errors, and gave a clear speedup.

After that, we saw that χ² mean and median were wrong. We traced the problem to a small algebra mistake in our vectorization: we had scaled the **columns** of `U[:nf, :nf]`, but the correct formula scales the **rows** (`diag(filt_vec) @ U`). We changed the code to row scaling (`filt_vec[:, None] * U[:nf, :nf]`), rebuilt the `(nt × nf)` core as before, and restored the χ² values to the baseline while keeping the speed benefits.

With correctness back, we listed safe next steps that keep the same interface and numbers: use reciprocal-based whitening to reduce divisions and allocations, preallocate and reuse the `(nt × nf)` work buffer across positivity iterations, cache invariants like `sa2` and `sb2` outside the loop, and make small improvements in `elogt` (reuse a precomputed max and threshold). These are low risk, easy to implement, and follow the same vectorization approach that worked here.
