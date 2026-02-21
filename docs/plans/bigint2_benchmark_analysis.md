# BigInt2 Benchmark Results & Optimization Roadmap

> **Benchmark location:** `benches/bigint/` (unified folder for BigInt10 vs BigInt2
> comparisons). Run with `pixi run bint` (interactive) or `pixi run bench bigint <op>`.
> BigUInt-only benchmarks remain in `benches/biguint/`.

## Benchmark Summary (2026-02-20, macOS arm64, Apple Silicon)

All benchmarks compare **BigInt2** (base-2^32) against **BigInt10/BigUInt** (base-10^9)
and **Python int** (CPython 3.13, GMP-backed). Speedup = Python time / Mojo time.
Values >1× mean faster than Python; <1× mean slower than Python.

**Key optimizations applied:**

- ✅ **PR0**: Fixed sqrt correctness bug (Newton converges from above + precision-doubling algorithm)
- ✅ **PR1**: Karatsuba multiplication (O(n^1.585)) with pointer-based inner loops and offset-based assembly
- ✅ **PR4a**: SIMD-optimized `parse_numeric_string` (two-pass architecture + `vectorize`-based digit extraction)
- ✅ **PR4c**: from_string micro-optimizations (fast paths, pre-allocation, raw pointers, balanced D&C split)
- ✅ **PR4d**: to_string micro-optimizations (fast paths for 1–2 words, `unsafe_ptr()`, byte buffer output)

### Overall Results (Average Across All Cases)

| Operation        | BigInt2 vs Python | BigInt10/BigUInt vs Python | BigInt2 vs BigInt10 |
| ---------------- | :---------------: | :------------------------: | :-----------------: |
| **Addition**     |     **4.30×**     |           2.36×            |    ~1.8× faster     |
| **Multiply**     |     **3.98×**     |           1.93×            |    ~2.1× faster     |
| **Floor Divide** |       1.24×       |           1.95×            |    ~0.6× slower     |
| **Trunc Divide** |       1.50×       |           2.32×            |    ~0.6× slower     |
| **Left Shift**   |     **4.97×**     |            N/A             |         N/A         |
| **Power**        |   **11.17×** ★    |           0.58×            |    ~19.3× faster    |
| **Sqrt**         |     **1.39×**     |      1.29× (BigUInt)       |    ~1.1× faster     |
| **from_string**  |     **1.42×**     |           6.57×            |    ~0.2× slower     |
| **to_string**    |     **6.03×**     |         **9.74×**          |    ~0.6× slower     |

★ Power average dominated by 2^N shift fast path (up to 140×). General cases: 0.65–0.94×.

### Key Findings

1. **Multiplication is now fast at ALL sizes.** With Karatsuba (O(n^1.585)) and
   pointer-based inner loops, BigInt2 beats Python even at 10000×10000 digits
   (1.34×). Previously 0.36× with schoolbook-only. The Karatsuba threshold is
   48 words (~460 digits), with three asymmetric cases handling unequal operand sizes.

2. **Power has a 2^N fast path.** Base-2 powers use left_shift instead of
   multiply, giving 5–140× speedup. For 2^32768 (~9864 digits): BigInt2 340 ns
   vs Python 47580 ns = **140× faster**. General non-2^N cases: 0.65–0.94×,
   limited by per-operation overhead in the square-and-multiply loop.

3. **Sqrt is now correct AND competitive at scale.** The correctness bug (wrong
   results at 1000+ digits) is fixed. A hybrid algorithm is used:
   - Hardware sqrt for ≤2 words (5–6.5× Python)
   - Newton's method for 3–54 words (~520 digits)
   - CPython-style precision-doubling for >54 words (O(M(n)) total work)
   At 10000 digits: 1.27× Python. Medium sizes (20–500 digits) remain 0.12–0.24×
   due to per-operation overhead of BigInt2 divisions.

4. **Division remains the primary bottleneck.** Still schoolbook O(n²). This
   limits sqrt, power, and to_string performance at medium sizes. BigInt2's
   division (~1.24×) is barely faster than Python. Knuth Algorithm D or
   Burnikel-Ziegler would be the next major win.

5. **to_string is greatly improved.** With PR4d fast paths, small numbers
   (≤20 digits) now achieve 15–29× Python. At 2000+ digits, D&C base conversion
   beats Python (1.0–1.38×). Medium range (100–1000 digits) remains 0.57–0.79×,
   limited by O(n²) simple-path division. BigInt10 still dominates at scale
   (34.5× at 10000 digits) due to trivial base-10^9 conversion.

6. **from_string is competitive.** 3.5× at small sizes, stays above 1× through
   10000 digits. D&C conversion (PR4b, entry at >10000 digits) improved 20K
   from 0.63→0.82× and 50K from 0.47→0.84×. The remaining gap at 20K+ digits
   is due to Karatsuba vs GMP multiplication (requires PR8 Toom-Cook/NTT).

7. **Shift is excellent.** Average 4.97× across all sizes. Degrades at extreme
   sizes (0.49× for 1 << 100000) due to memory allocation overhead.

---

### Detailed Per-Case Observations

**Addition by size** (unchanged from initial benchmark):

| Size            | BigInt2 vs Python | BigInt10 vs Python |
| --------------- | :---------------: | :----------------: |
| Small (<20 dig) |       5–8×        |       2.5–4×       |
| 500 digits      |       3.71×       |       1.59×        |
| 1000 digits     |       2.45×       |       1.43×        |
| 5000 digits     |       1.59×       |       1.27×        |
| 10000 digits    |       1.54×       |       1.91×        |

At 10000 digits, BigInt10 overtakes BigInt2 in addition. This is because BigUInt
(underlying BigInt10) uses SIMD vectorization for addition at large sizes.

**Multiplication by size** (post-Karatsuba):

| Size              | BigInt2 (ns) | Python (ns) | BigInt2 vs Python | BigInt10 vs Python |
| ----------------- | -----------: | ----------: | :---------------: | :----------------: |
| Small (<20 dig)   |        40–50 |     230–310 |     4.6–7.8×      |       2.6–6×       |
| 50 × 50 dig       |          100 |         300 |       3.0×        |       1.88×        |
| 100 × 100 dig     |          160 |         530 |       3.3×        |       1.10×        |
| 200 × 200 dig     |          540 |       1,000 |       1.85×       |       0.74×        |
| 300 × 300 dig     |          680 |       1,050 |       1.55×       |       0.54×        |
| 500 × 500 dig     |        1,320 |       1,940 |       1.47×       |       0.46×        |
| 600 × 600 dig     |        1,720 |       2,810 |       1.63×       |       0.55×        |
| 700 × 700 dig     |        2,130 |       2,790 |       1.31×       |       0.45×        |
| 1000 × 1000 dig   |        4,660 |       5,650 |       1.21×       |       0.43×        |
| 2000 × 2000 dig   |       15,160 |      16,740 |       1.10×       |       0.41×        |
| 5000 × 5000 dig   |       63,580 |      84,910 |       1.34×       |       0.57×        |
| 10000 × 10000 dig |      194,720 |     261,370 |       1.34×       |       0.57×        |

**Critical improvement:** BigInt2 is now faster than Python at **every** size.
Previously, 2000+ digit multiplication was 0.36–0.57× Python. Karatsuba brought
10000-digit multiply from 745 µs → 195 µs (3.8× internal speedup).

**Floor Division by size** (✅ optimized with slice-based Burnikel-Ziegler):

| Size              | BigInt2 vs Python | BigInt10 vs Python |
| ----------------- | :---------------: | :----------------: |
| Small (<20 dig)   |     1.0–1.5×      |        2–4×        |
| 500/200 digits    |     1.5–1.7×      |        0.3×        |
| 2000/1000 digits  |       ~1.3×       |        0.2×        |
| 5000/2500 digits  |       ~1.8×       |        0.4×        |
| 10000/5000 digits |       ~1.4×       |        0.4×        |

BigInt2 division now beats Python at all sizes thanks to slice-based B-Z.
Average across 62 benchmark cases: **1.14× Python**.

**Power by case** (post-optimization):

| Case                  | Result digits | BigInt2 (ns) | Python (ns) | BigInt2 vs Python |
| --------------------- | :-----------: | -----------: | ----------: | :---------------: |
| 2^10                  |       4       |          100 |         340 |       3.4×        |
| 2^64                  |      20       |           80 |         280 |       3.5×        |
| 2^256                 |      78       |           80 |         360 |       4.5×        |
| 2^1024                |      309      |          100 |         500 |       5.0×        |
| 2^2048                |      617      |          100 |         840 |       8.4×        |
| 2^8192                |     2467      |          160 |       8,460 |     **52.9×**     |
| 2^32768               |     9865      |          340 |      47,580 |    **140.0×**     |
| 10^20                 |      21       |          340 |         300 |       0.88×       |
| 10^100                |      101      |          520 |         480 |       0.92×       |
| 10^500                |      501      |        1,200 |         820 |       0.68×       |
| 10^2000               |     2001      |        8,020 |       6,420 |       0.80×       |
| 10^5000               |     5001      |       38,960 |      29,000 |       0.74×       |
| 3^100                 |      48       |          460 |         400 |       0.87×       |
| 7^50                  |      43       |          420 |         380 |       0.90×       |
| 99^50                 |      100      |          460 |         360 |       0.78×       |
| 99^500                |      998      |        3,140 |       2,080 |       0.66×       |
| 99^2500               |     4990      |       43,660 |      28,360 |       0.65×       |
| 1000^100              |      301      |          640 |         600 |       0.94×       |
| Large base, small exp |      81       |          340 |         300 |       0.88×       |
| (-2)^100              |      31       |           80 |         320 |       4.0×        |
| (-2)^101              |      32       |          100 |         320 |       3.2×        |

**Key observations:**

- 2^N cases: 3.4–140× (uses left_shift, O(1) per bit instead of multiplies)
- Small general (result <200 digits): 0.78–0.94× (competitive with Python)
- Large general (result >1000 digits): 0.65–0.80× (per-multiply overhead accumulates)
- Eliminating the wasted final squaring gave a modest boost across all cases
- BigInt2 always beats BigInt10 for power (BigInt10 averages 0.58×)

**Sqrt by size** (post-precision-doubling fix):

| Size               | BigInt2 (ns) | Python (ns) | BigInt2 vs Python | BigUInt vs Python |
| ------------------ | -----------: | ----------: | :---------------: | :---------------: |
| Small (≤7 dig)     |        40–60 |     260–300 |     5.0–6.5×      |     4.3–5.0×      |
| 20 digits          |        2,840 |         340 |       0.12×       |       0.33×       |
| 50 digits          |        2,020 |         480 |       0.24×       |       0.23×       |
| 100 digits         |        3,440 |         560 |       0.16×       |       0.11×       |
| 200 digits         |        4,680 |         880 |       0.19×       |       0.06×       |
| 500 digits         |        8,860 |       1,880 |       0.21×       |       0.06×       |
| **1000 digits** ✅  |        7,400 |       4,340 |     **0.59×**     |       0.04×       |
| **2000 digits** ✅  |       15,520 |      14,540 |     **0.94×**     |       0.03×       |
| **5000 digits** ✅  |       62,840 |      74,980 |     **1.19×**     |       0.05×       |
| **10000 digits** ✅ |      204,140 |     259,280 |     **1.27×**     |       0.05×       |

✅ All results are now **correct** (verified `sqrt(n)^2 <= n < (sqrt(n)+1)^2`).

**Key observations:**

- Small (≤2 words): Hardware sqrt, 5–6.5× Python
- Medium (20–500 digits): Newton's method, 0.12–0.24× (division overhead dominates each iteration)
- Large (1000+ digits): Precision-doubling algorithm, approaching and surpassing Python
- The crossover to beating Python is at ~2000 digits
- BigUInt is extremely slow at all sizes >50 digits (schoolbook division in its Newton iterations)
- Fixing division (PR2) would dramatically improve the medium-size sqrt performance

**from_string by size** (re-benchmarked with SIMD `parse_numeric_string`):

| Size         | BigInt2 (before) | BigInt2 (after PR4a) | BigInt10 vs Python |
| ------------ | :--------------: | :------------------: | :----------------: |
| 2 digits     |      3.80×       |      **4.20×**       |       1.91×        |
| 9 digits     |      2.67×       |      **3.40×**       |       1.06×        |
| 20 digits    |      2.12×       |        2.11×         |       1.58×        |
| 50 digits    |      1.50×       |        1.08×         |       1.80×        |
| 100 digits   |      1.05×       |      **1.24×**       |       1.13×        |
| 200 digits   |      0.94×       |      **1.00×**       |       1.03×        |
| 500 digits   |      1.28×       |        1.12×         |       1.24×        |
| 1000 digits  |      1.12×       |        1.14×         |       1.92×        |
| 2000 digits  |      0.78×       |      **1.21×**       |       4.04×        |
| 5000 digits  |      1.08×       |        1.17×         |       8.90×        |
| 10000 digits |      1.34×       |        1.35×         |       19.93×       |
| 20000 digits |      0.63×       |      **0.85×**       |       21.25×       |
| 50000 digits |      0.47×       |      **0.86×**       |       33.38×       |

**Average BigInt2 vs Python:** 1.42× → **1.57×** (14 cases, +11%)

The SIMD optimization of `parse_numeric_string` provides the most visible benefit
at small sizes (2–9 digits, +10–27%) where parsing is a large fraction of total
time. At large sizes (1000+), the O(n²) base-10 → base-2^32 conversion dominates.

**to_string by size:**

| Size         | BigInt2 vs Python | Before PR4d | BigInt10 vs Python |
| ------------ | :---------------: | :---------: | :----------------: |
| 2 digits     |     **29.0×**     |    3.3×     |       19.3×        |
| 9 digits     |     **20.0×**     |    2.2×     |       15.4×        |
| 20 digits    |     **15.5×**     |    1.2×     |        5.0×        |
| 50 digits    |     **1.0×**      |    0.7×     |        1.2×        |
| 100 digits   |       0.79×       |    0.5×     |        0.9×        |
| 200 digits   |       0.71×       |    0.5×     |        1.0×        |
| 500 digits   |       0.57×       |    0.5×     |        1.7×        |
| 1000 digits  |       0.57×       |    0.6×     |        4.0×        |
| 2000 digits  |     **1.0×**      |    0.6×     |        8.6×        |
| 5000 digits  |     **1.38×**     |    0.6×     |       24.4×        |
| 10000 digits |     **1.17×**     |    0.4×     |       34.5×        |

**PR4d gains:** Fast paths for 1-word and 2-word values give 15–29× at ≤20 digits.
D&C base conversion (PR3) makes 2000+ digits beat Python. Medium range (100–1000)
remains 0.57–0.79×, limited by O(n²) repeated division in the simple path.
BigInt10's to_string advantage grows with size (trivial in base-10^9).

**Left Shift by size:**

| Case               | BigInt2 vs Python |
| ------------------ | :---------------: |
| 1 << 1             |       6.6×        |
| 1 << 32            |       5.4×        |
| 1 << 64            |       6.6×        |
| 1 << 256           |       6.5×        |
| 1 << 1024          |       5.9×        |
| 100-digit << 100   |       6.2×        |
| 200-digit << 256   |       5.8×        |
| 1000-digit << 1000 |       4.3×        |
| 5000-digit << 5000 |       1.8×        |
| 10000-dig << 10000 |       1.4×        |
| 1 << 100000        |       0.5×        |
| 1000-dig << 32768  |       1.2×        |

Shift is fast for typical sizes (4–7×). Degrades at extreme sizes due to
memory allocation overhead for very large result buffers.

---

## Resolved Bugs

### ✅ BigInt2 sqrt correctness bug at 1000+ digits (FIXED)

**Symptom:** sqrt(10^999) returned 975387... instead of the correct 999...9.
Newton's method converged to the wrong answer because the initial guess was an
underestimate, allowing convergence to the wrong quadratic residue.

**Fix:** Two-part solution:

1. **Newton's method (≤54 words):** Changed initial guess to always be an
   **overestimate** using hardware sqrt of the top 1–2 words with ceiling
   rounding. Newton's method then converges monotonically from above.
2. **Precision-doubling (>54 words):** Implemented CPython's precision-doubling
   algorithm that starts with a 1-bit seed and doubles precision each iteration.
   Each iteration at precision p does a division of size 2p, so total work is
   O(M(n)) instead of O(M(n) * log n).

**Verification:** All 16 sqrt benchmark cases produce correct results. Separately
tested correctness up to 5000 digits with `sqrt(n)^2 <= n < (sqrt(n)+1)^2`.

---

## Optimization Roadmap

### ✅ PR 0 (BUGFIX): Fix BigInt2 sqrt correctness — DONE

Fixed with overestimate-based Newton's method + CPython precision-doubling algorithm.
See "Resolved Bugs" section above.

### ✅ PR 1: Karatsuba Multiplication — DONE

Implemented Karatsuba O(n^1.585) in `_multiply_magnitudes_karatsuba()` with:

- **Threshold:** CUTOFF_KARATSUBA = 48 words (~460 digits)
- **Three Karatsuba cases:** normal (balanced), asymmetric Case 1 (a shorter), and
  asymmetric Case 2 (b shorter)
- **Pointer-based inner loops:** `_data` pointer access in schoolbook avoids bounds checking
- **Offset-based assembly:** `_add_at_offset_inplace(a, b, offset)` replaces the
  expensive `shift_left_words + add` pattern, eliminating O(n) memory copies
- **Slice-based sub-operations:** `_multiply_magnitudes_school(a, a_start, a_end, b, b_start, b_end)`
  avoids creating sub-lists for Karatsuba's recursive calls

**Result:** 10000-digit multiply: 745 µs → 195 µs (**3.8× internal speedup**).
BigInt2 now beats Python at ALL multiplication sizes.

---

### PR 2: Fast Division (Knuth Algorithm D / Burnikel-Ziegler)

**Priority: HIGHEST** — The most impactful remaining optimization

**Current:** `_divmod_magnitudes()` uses Knuth's Algorithm D (schoolbook).
BigInt2 is barely faster than Python at medium sizes and slower at 10000 digits (0.88×).
Division is the bottleneck for:

- **Sqrt medium sizes** (20–500 digits): per-iteration division overhead makes BigInt2
  0.12–0.24× Python despite correct algorithm
- **Power at scale** (99^2500 = 0.65×): many multiplies done correctly, but the
  occasional division in Newton's method slows sqrt
- **to_string** (O(n²) repeated division by 10^9)

**First attempt (Burnikel-Ziegler, copy-based):** Implemented B-Z with recursive
`_bz_div_two_by_one` / `_bz_div_three_by_two`, but benchmarks showed regressions
at 700+ digits (0.39–0.77× Python vs 0.88–1.47× with schoolbook). The root cause
is excessive `List[UInt32]` allocation in recursive calls — each level creates
multiple copies via `_get_words_slice`, `_shift_left_words_inplace`, and
`_add_magnitudes_inplace`. The code is retained but the dispatch is disabled.

At 300–600 digits, B-Z did show real wins (1.3–1.7× Python) thanks to shallow
recursion depth, confirming the algorithm IS faster — the implementation just
needs to minimize allocations.

**Second attempt (slice-based B-Z): ✅ DONE.** Rewrote B-Z following BigUInt's
proven approach — passes `(list, start, end)` bounds through the recursion instead
of materializing sub-lists. Key optimizations:

1. **Slice-based recursion**: `_bz_two_by_one_slices` and `_bz_three_by_two_slices`
   pass bounds through to avoid copying until the Knuth D base case.
2. **Prenormalized base case**: `_divmod_knuth_d_from_slices` operates directly on
   pre-normalized slices, reads divisor via pointer offset (no v copy), reducing
   copies from 5 to 1 per base case call.
3. **Minimal padding**: Instead of rounding divisor to 2^k × cutoff (97% waste for
   520 words → 1024), rounds to the next even number. The recursion handles odd
   sizes by falling through to the base case.
4. **Optimized `_shift_left_words_inplace`**: backward pointer copy instead of
   temp buffer allocation.
5. **Pre-allocated quotient**: top-level uses `_add_at_offset_inplace` to place
   each quotient digit, avoiding expensive repeated `_shift_left_words_inplace`.
6. **Helper functions**: `_add_from_slice_inplace`, `_multiply_magnitudes_slices`,
   `_is_zero_in_range`, `_decrement_inplace` — all avoid unnecessary copies.

**Cutoff tuning**: `CUTOFF_BURNIKEL_ZIEGLER = 64` (words) gave the best results.
Tested 32, 64, and 128. Cutoff=32 triggered B-Z too early (overhead > gain for
300–600 digit divisors). Cutoff=128 delayed B-Z too long.

**Benchmark results (slice-based B-Z, cutoff=64, minimal padding):**

| Size                 | Schoolbook (vs Python) | B-Z slice-based (vs Python)  |
| -------------------- | ---------------------- | ---------------------------- |
| 300 dig / 200 dig    | 1.48×                  | 1.48× (schoolbook, < cutoff) |
| 500 dig / 200 dig    | 1.55×                  | 1.55× (schoolbook, < cutoff) |
| 600 dig / 300 dig    | 1.2×                   | 1.2× (schoolbook, < cutoff)  |
| 700 dig / 350 dig    | —                      | ~1.2× (B-Z, was 0.39×)       |
| 2000 dig / 1000 dig  | —                      | ~1.3× (B-Z, was 0.69×)       |
| 5000 dig / 2500 dig  | —                      | ~1.8× (B-Z, was 0.84×)       |
| 10000 dig / 5000 dig | 0.88×                  | **~1.4×** (B-Z, was 0.77×)   |

**Average across 62 test cases: ~1.14× Python** (dominated by small-number constant
overhead). For B-Z cases (divisor > 600 digits): consistently 1.2–1.8× Python.

**Status: ✅ Complete.** All 60 tests pass, B-Z dispatch enabled, code merged.

**Remaining opportunities (future work):**

- Barrett division (reciprocal via Newton) for very large balanced divisions
- SIMD-ized Knuth D base case for additional constant-factor improvement
- GMP-style asymmetric division for highly unbalanced operands (m >> n)

---

### PR 3: Optimized to_string (Divide-and-Conquer Base Conversion)

**Priority: HIGH** — BigInt2's biggest weakness vs BigInt10 (31.5× gap at 10000 digits)

**Status: ✅ DONE** (2026-02-21)

**Implementation:**

Divide-and-conquer base conversion replaces the O(n²) `to_bigint10()` path:

1. Precompute a power table: `powers[k] = 10^(2^k)` as BigInt2 values
2. Find the largest `k` where `powers[k] ≤ n`
3. `divmod(n, powers[k])` → split into high and low halves
4. Recursively convert each half; zero-pad low part to exactly `2^k` digits
5. Base case: simple repeated division by 10^9 for small sub-problems

Key optimizations:

- **Dual threshold**: entry threshold = 128 words (only enter D&C when B-Z
  division will help); base-case threshold = 64 words (within recursion)
- **Lazy power table**: only build `powers[0..max_level-1]`, skipping the
  unused largest entry (saves one expensive squaring)
- **Leverages B-Z division**: the sub-quadratic Burnikel-Ziegler algorithm
  from PR2 makes the recursive divisions fast

**Results (vs Python `int.__str__()`):**

| Size (digits) | Before D&C | After D&C | Improvement                       |
| ------------- | ---------- | --------- | --------------------------------- |
| 500           | 0.51×      | 0.56×     | (simple path, unchanged)          |
| 1000          | 0.53×      | 0.57×     | no D&C overhead (entry threshold) |
| 2000          | 0.86×      | 0.94×     | 1.1× faster                       |
| 5000          | 1.06×      | **1.38×** | D&C + B-Z benefit                 |
| 10000         | 0.88×      | **1.16×** | D&C + B-Z benefit                 |

The "before D&C" column already includes B-Z division from PR2.
At 5000+ digits, BigInt2 to_string now **beats Python**.

**Remaining gap at medium sizes (100–1000 digits):** The simple O(n²) method
is limited by per-word UInt64 division. Python uses GMP's assembly-optimized
routines. See PR4d for micro-optimizations that partially close this gap.

---

### PR 4: Optimized from_string (Divide-and-Conquer)

**Priority: MEDIUM** — Already 1.57× avg but degrades to 0.85× at 20000 digits

#### ✅ PR 4a: SIMD-optimized `parse_numeric_string` — DONE

Rewrote the string-to-digit-array parser in `str.mojo` with:

- **Two-pass architecture:** Pass 1 scans structure (sign, decimal point, exponent,
  separators) with position tracking. Pass 2 extracts digit values via SIMD.
- **Three extraction paths:**
  - **Fast path** (pure digits): `vectorize[16]()` processes 16 bytes at a time,
    subtracting `ord("0")` via SIMD `load`/`store` on `List[UInt8]._data`
  - **Medium path** (digits + decimal point): two `vectorize` calls around the `.`
  - **Slow path** (with `_` separators): byte-by-byte extraction
- **`vectorize` with `unified` capture:** Uses `unified {mut coef, read value_bytes}`
  for proper Mojo origin tracking instead of manual SIMD width cascades

**Result:** BigInt2 from_string average 1.42× → **1.57×** (+11%). Best improvement
at small sizes: 2 digits 3.8→4.2× (+10%), 9 digits 2.67→3.40× (+27%).

#### ✅ PR 4b: Divide-and-Conquer Base Conversion — DONE

Implemented D&C decimal→binary conversion in `_from_decimal_digits_dc()`:

1. Precompute a power table: `powers[k] = 10^(2^k)` as BigInt2 values
2. Recursively split: `high = digits[0..mid] * 10^(n-mid) + digits[mid..n]`
3. Base case: switch to simple `_from_decimal_digits_simple()` at 256 digits
4. Entry threshold: only enter D&C when digit_count > 10000

#### ✅ PR 4c: from_string Micro-optimizations — DONE

Optimized both the simple and D&C paths for from_string:

**Simple path (`_from_decimal_digits_simple`) optimizations:**

- Fast paths for ≤ 9 digits (single UInt32 word, zero allocation overhead)
  and 10–19 digits (UInt64 → 1–2 words)
- Pre-allocates the full word array to maximum size, avoiding all dynamic
  growth (append / realloc) during conversion
- Handles the first (possibly shorter) chunk separately so the hot loop
  always processes exactly 9 digits with a compile-time 10^9 constant
- Raw pointer access (`_data`) for both digit and word arrays, eliminating
  bounds-checking in the inner multiply-add loop
- Tracks live word count in a local variable; single trim pass at end

**D&C path (`_from_decimal_digits_dc`) optimizations:**

- **Balanced splitting**: splits at the largest 2^k ≤ digit_count/2 instead
  of the largest 2^k < digit_count. This keeps both halves reasonably close
  in size (within ~3:1 worst case), producing balanced operands for Karatsuba.
- **Smaller power table**: builds only up to `floor(log2(digit_count/2))`
  entries instead of `ceil(log2(digit_count))`, saving one expensive
  top-level squaring (e.g., skipping 10^32768 for 50K digits).
- **Smarter max_level propagation**: the high recursive call receives
  `level` (not `level-1`) since high ≥ digit_count/2 and may need the
  same power-of-2 boundary. The low call receives `level-1`.
- Combine step uses `_add_magnitudes_inplace` for true in-place add

**Result (from_string speedup vs Python `int(string)`):**

| Digits | Before PR4c | After PR4c | Improvement |
| ------ | ----------- | ---------- | ----------- |
| 2      | 3.67×       | 3.83×      | +4%         |
| 9      | 3.60×       | 3.50×      | ~same       |
| 20     | 2.38×       | 2.63×      | +10%        |
| 50     | 1.33×       | 2.40×      | +80%        |
| 100    | 1.19×       | 1.93×      | +62%        |
| 200    | **0.94×**   | **1.10×**  | **FIXED**   |
| 500    | 1.18×       | 1.24×      | +5%         |
| 1000   | 1.06×       | 1.10×      | +4%         |
| 2000   | 1.04×       | 1.11×      | +7%         |
| 5000   | 1.13×       | 1.12×      | ~same       |
| 10000  | 1.16×       | 1.06×      | ~same       |
| 20000  | **0.83×**   | **0.98×**  | **+18%**    |
| 50000  | **0.81×**   | **1.00×**  | **+23%**    |
| Avg    | **1.53×**   | **1.78×**  | **+16%**    |

All sizes now ≥ 0.98× Python. The remaining tiny gap at 20K digits
is due to Karatsuba O(n^1.585) vs Python/GMP's Toom-Cook algorithms.

---

#### ✅ PR 4d: to_string Micro-optimizations — DONE

Optimized `_magnitude_to_decimal_simple` and the D&C string-building path:

**Simple path (`_magnitude_to_decimal_simple`) optimizations:**

- **Fast path for 1-word values** (≤10 digits): `return String(Int(words[0]))` —
  delegates entirely to native int-to-string, zero BigInt2 overhead
- **Fast path for 2-word values** (≤20 digits): combines into UInt64 and uses
  `return String(combined_value)` — single native conversion
- **Raw pointer inner loop**: `var dp = dividend.unsafe_ptr()` for O(1) access
  to dividend words during repeated division, eliminating bounds-check overhead
  (NOTE: `List._data` causes heap corruption — only `List.unsafe_ptr()` is safe)
- **`div_len` tracking**: tracks the number of significant words in the dividend
  via a local variable, shrinking the division loop naturally instead of calling
  `shrink()` or scanning for leading zeros after each division pass
- **Byte buffer output**: extracts 9 digits per chunk into an `InlineArray[UInt8, 9]`
  on the stack, appends to a `List[UInt8]` buffer, then constructs the final
  string with `String(unsafe_from_utf8=buf^)` — avoids all string concatenation

**D&C path optimizations:**

- **Power table**: builds entries via `power_table[k-1] * power_table[k-1]`
  without `.copy()` — Mojo's value semantics handle this correctly
- **Recursive string building**: replaced `result += "0" * pad_needed` with
  byte buffer approach: append pad zeros + low_str bytes to `List[UInt8]`,
  construct via `String(unsafe_from_utf8=buf^)`

**What was tried and rejected:**

- **UInt128 chunks with 10^18 divisor**: 128-bit division/modulo is software-
  emulated on ARM64 (no hardware instruction). Made medium numbers 2× SLOWER.
- **Lower D&C thresholds (entry=24, base=8)**: D&C overhead (power table
  computation + BigInt2 division) exceeded simple-path cost for 500–10000 digits.
  Restored original thresholds (entry=128, base=64).

**Result (to_string speedup vs Python `str(int_value)`):**

| Size         | Before PR4d | After PR4d | Improvement            |
| ------------ | ----------- | ---------- | ---------------------- |
| 2 digits     | 3.3×        | **29.0×**  | **+780%** (fast path)  |
| 9 digits     | 2.2×        | **20.0×**  | **+810%** (fast path)  |
| 20 digits    | 1.2×        | **15.5×**  | **+1190%** (fast path) |
| 50 digits    | 0.7×        | **1.0×**   | **+43%**               |
| 100 digits   | 0.5×        | 0.79×      | +58%                   |
| 200 digits   | 0.5×        | 0.71×      | +42%                   |
| 500 digits   | 0.5×        | 0.57×      | +14%                   |
| 1000 digits  | 0.6×        | 0.57×      | ~same                  |
| 2000 digits  | 0.6×        | **1.0×**   | **+67%**               |
| 5000 digits  | 0.6×        | **1.38×**  | **+130%**              |
| 10000 digits | 0.4×        | **1.17×**  | **+193%**              |
| Avg          | 0.97×       | **6.03×**  | **+522%**              |

Fast paths dominate the average. Excluding ≤20-digit fast paths, the average
for 50–10000 digits improved from 0.54× to **0.91×** (+69%).

---

### PR 5: True In-Place Arithmetic Operations ✅ DONE

**Priority: MEDIUM-HIGH** — Performance infrastructure that benefits all callers

**Status:** All 11 `__i*__` dunders now call true in-place functions. Fully tested.

**Previous state:** All augmented assignment operators (`+=`, `-=`, `*=`, `//=`,
`%=`, `<<=`, `>>=`, `&=`, `|=`, `^=`) created a new BigInt2 and replaced `self`:

```mojo
fn __iadd__(mut self, other: Self):
    self = arithmetics.add(self, other)  # allocates new BigInt2
```

**Implementation (two phases):**

**Phase 1 — Core arithmetic (arithmetics.mojo):**

- `add_inplace(mut x, other)` — true in-place `+=`. Uses existing
  `_add_magnitudes_inplace` / `_subtract_magnitudes_inplace` with sign handling
  (same sign → add magnitudes; different signs → subtract smaller from larger)
- `add_inplace_int(mut x, value: Int)` — optimized `+= Int`. Builds at most
  2 words from the Int value, avoids BigInt2 construction entirely
- `subtract_inplace(mut x, other)` — true in-place `-=`. Flips other's sign
  conceptually and applies add logic
- `multiply_inplace(mut x, other)` — computes `_multiply_magnitudes`, moves
  result words into x (can't be truly in-place due to read-during-write, but
  avoids full BigInt2 construction)
- Helper: `_compare_magnitudes_words(a, b) -> Int8` for raw word-list comparison

**Phase 2 — Shifts, division, bitwise (arithmetics.mojo + bitwise.mojo):**

- `left_shift_inplace(mut x, shift)` — builds shifted word list and moves into x
- `right_shift_inplace(mut x, shift)` — shifts words in-place, truncates, handles
  arithmetic right shift for negative numbers (round toward -∞)
- `floor_divide_inplace(mut x, other)` — computes `_divmod_magnitudes`, moves
  quotient words into x with correct sign/rounding
- `floor_modulo_inplace(mut x, other)` — computes `_divmod_magnitudes`, moves
  remainder words into x with correct sign adjustment
- `bitwise_and_inplace(mut a, b)` — parametric `_binary_bitwise_op_inplace["and"]`.
  Non-negative fast path: AND in-place, truncate to `min_len`
- `bitwise_or_inplace(mut a, b)` — extend a if shorter, OR in-place
- `bitwise_xor_inplace(mut a, b)` — extend a if shorter, XOR in-place

**Internal callers updated:**

- `_sqrt_newton()`: `guess -= BigInt2(1)` and `guess += BigInt2(1)` now use
  true in-place ops instead of allocating `guess = guess - BigInt2(1)`
- `_sqrt_precision_doubling()`: `a -= BigInt2(1)` for final adjustment

**Testing:**

- 41 tests in `test_bigint2_inplace.mojo`: `+=`, `-=`, `*=` (basic, zero, sign,
  carry, large, consistency with non-inplace, accumulator loops, factorial)
- 59 tests in `test_bigint2_inplace_extended.mojo`: `<<=`, `>>=`, `//=`, `%=`,
  `&=`, `|=`, `^=` (basic, edge cases, negative, large, consistency with
  non-inplace, chained operations, mixed-op chains)
- All 201 BigInt2 tests pass

**Expected performance impact:**

| Scenario                                           | Savings                                                   | Estimated speedup |
| -------------------------------------------------- | --------------------------------------------------------- | ----------------- |
| Small values (1–2 words), high-frequency `+=`      | 1 List alloc + BigInt2 construct per op                   | ~5–15%            |
| Accumulator loops (`sum += i` × 1000)              | 1000 allocations eliminated                               | ~10–20%           |
| Medium values (10–100 words)                       | Same savings, but arithmetic dominates                    | ~2–5%             |
| Large values (1000+ words)                         | Arithmetic time dominates completely                      | <1%               |
| `*=`, `//=`, `%=`                                  | Only saves BigInt2 construction (can't avoid temp buffer) | ~1–3%             |
| Bitwise `&=`, `\|=`, `^=` (non-negative fast path) | True in-place word modification                           | ~5–10%            |

---

### PR 6: Bitwise Operations (AND, OR, XOR, NOT) ✅ DONE

**Priority: MEDIUM** — Completes the integer API surface

**Status:** Implemented and fully tested.

**Implementation:**

- Created `bitwise.mojo` module with parametric `_binary_bitwise_op[op]` helper
- Python-compatible two's complement semantics for negative numbers
- Fast path for both non-negative operands (direct word-by-word, no TC conversion)
- General path: inline TC conversion (subtract 1 from magnitude, bitwise NOT)
- Dunders: `__and__`, `__or__`, `__xor__`, `__invert__`, `__iand__`, `__ior__`, `__ixor__`
- Each binary op has `(Self, Self)` and `(Self, Int)` overloads
- 31 tests: NOT, AND, OR, XOR (positive/negative/mixed), augmented assignment,
  De Morgan's laws, commutativity, identities, word-boundary values, Python cross-checks

---

### PR 7: GCD, Extended GCD, and Modular Arithmetic ✅ DONE

**Priority: MEDIUM** — Useful for cryptographic and number theory applications

**Status:** Implemented and fully tested.

**Implementation (number_theory.mojo):**

- `gcd(a, b)` — Binary GCD (Stein's algorithm): uses only subtraction and
  right-shifts, natural for base-2^32 representation. Helper
  `_count_trailing_zeros` operates at the word level for efficient power-of-2
  factoring.
- `extended_gcd(a, b)` → `(gcd, x, y)` — Iterative extended Euclidean
  algorithm. Returns Bézout coefficients satisfying `a*x + b*y = gcd`.
  Correctly handles negative inputs via sign adjustment.
- `lcm(a, b)` — Computed as `|a| / gcd(a,b) * |b|` (divides first to keep
  intermediates small).
- `mod_pow(base, exp, mod)` — Right-to-left binary exponentiation with modular
  reduction at each step. Intermediates never exceed `mod²`. Accepts both
  `BigInt2` and `Int` exponents. Handles negative base via floor modulo.
- `mod_inverse(a, mod)` — Modular multiplicative inverse via `extended_gcd`.
  Returns `x` in `[0, mod)` such that `a*x ≡ 1 (mod m)`. Raises when
  `gcd(a, mod) ≠ 1`.

**Testing:** 31 tests covering:

- GCD: zero/basic/negative/commutativity/powers-of-two/large-coprime/common-factor
- Extended GCD: known coefficients, Bézout identity verification, zero/negative
  inputs, multiple input pairs
- LCM: basic/zero/negative, `gcd * lcm = |a*b|` property
- mod_pow: basic/edge/Fermat's little theorem/large exponents/negative base/errors
- mod_inverse: known inverses, roundtrip `a * inv(a) ≡ 1`, non-existence errors

---

### PR 8: Reassign BInt alias from BigInt10 → BigInt2

**Priority: LOW** — Wait until BigInt2 is clearly better across the board

**Prerequisite:** PRs 2–3 complete, BigInt2 faster than BigInt10 in most operations.

---

### PR 9: Toom-Cook 3 / NTT Multiplication

**Priority: LOW** — Only valuable for very large numbers (10000+ digits)

**Prerequisite:** PR 1 (Karatsuba — done)

**Current status:** With Karatsuba, BigInt2 is already 1.34× Python at 10000
digits. Toom-Cook 3 (O(n^1.465)) would help at 50000+ digits. NTT for extreme
sizes (100000+).

---

## Summary: Priority Order

| PR   | Title                           | Status     | Priority    | Impact                     |
| ---- | ------------------------------- | ---------- | ----------- | -------------------------- |
| PR0  | Fix sqrt correctness bug        | ✅ **DONE** | CRITICAL    | correctness (fixed)        |
| PR1  | Karatsuba Multiplication        | ✅ **DONE** | HIGHEST     | mul 3.8× faster at scale   |
| PR2  | Fast Division (Knuth D + B-Z)   | ✅ **DONE** | HIGHEST     | div, sqrt, to_string       |
| PR3  | D&C to_string                   | ✅ **DONE** | HIGH        | to_string: 1.38× at 5K     |
| PR4a | SIMD parse_numeric_string       | ✅ **DONE** | MEDIUM      | from_string +11% avg       |
| PR4b | D&C from_string                 | ✅ **DONE** | MEDIUM      | from_string at scale       |
| PR4c | from_string micro-optimizations | ✅ **DONE** | MEDIUM      | all sizes ≥ 0.98× Python   |
| PR4d | to_string micro-optimizations   | ✅ **DONE** | MEDIUM      | to_string 0.97×→6.03× avg  |
| PR5  | True in-place iadd/isub/imul    | ✅ **DONE** | MEDIUM-HIGH | perf infra, readability    |
| PR6  | Bitwise AND/OR/XOR/NOT          | ✅ **DONE** | MEDIUM      | API completeness           |
| PR7  | GCD + Modular Arithmetic        | ✅ **DONE** | MEDIUM      | applications               |
| PR8  | Reassign BInt → BigInt2         | TODO       | LOW         | ergonomics                 |
| PR9  | Toom-Cook / NTT                 | TODO       | LOW         | extreme sizes (50000+ dig) |
