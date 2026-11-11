"""Finite checks for "Spinors of exceptional super Yang-Mills theories are not brane cohomologies".

Run: python3 verify_cohomology.py   (needs numpy and scipy)
Every check prints PASS or FAIL; the last line gives the totals.
"""
import itertools
from fractions import Fraction as F
from math import comb
import numpy as np
import scipy.sparse as sp

results = []
def check(name, ok):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name)

# 1. Binomial sums of eqs. (3.4)-(3.5) of arXiv:1906.10709
odd = [comb(15, k) for k in range(1, 16, 2)]
even = [comb(15, k) for k in range(0, 15, 2)]
check("odd forms on R^15: 15,455,3003,6435,5005,1365,105,1 sum to 16384",
      odd == [15, 455, 3003, 6435, 5005, 1365, 105, 1] and sum(odd) == 2**14)
check("even forms on R^15 sum to 16384, last entry C(15,14)=15", sum(even) == 2**14 and even[-1] == 15)
check("the list printed in (3.5), ending in 25, sums to 16394",
      sum([1, 105, 1365, 5005, 6435, 3003, 455, 25]) == 16394)

# 2. Irreducible representations of so(2m+1) of small dimension (Weyl dimension formula)
def irreps_B(r, maxdim, maxc=2):
    pos = []
    for i in range(r):
        for j in range(i + 1, r):
            for s in (1, -1):
                v = [0] * r; v[i] = 1; v[j] = s; pos.append(v)
        v = [0] * r; v[i] = 1; pos.append(v)
    rho = [F(2 * r - 1 - 2 * i, 2) for i in range(r)]
    fw = [[1] * (i + 1) + [0] * (r - i - 1) for i in range(r - 1)] + [[F(1, 2)] * r]
    out = []
    for c in itertools.product(range(maxc + 1), repeat=r):
        lam = [sum(ci * fw[i][k] for i, ci in enumerate(c)) for k in range(r)]
        num = den = F(1)
        for a in pos:
            num *= sum((lam[k] + rho[k]) * a[k] for k in range(r))
            den *= sum(rho[k] * a[k] for k in range(r))
        if num / den <= maxdim:
            out.append(int(num / den))
    return sorted(out)
check("so(7): irreducible dimensions <= 16 are 1, 7, 8", irreps_B(3, 16) == [1, 7, 8])
check("so(11): irreducible dimensions <= 24 are 1, 11", irreps_B(5, 24) == [1, 11])
check("so(15): irreducible dimensions <= 32 are 1, 15", irreps_B(7, 32) == [1, 15])
check("2^m > 4m+4 and m(2m+1) > 4m+4 for m = 2n+1, n = 2..40",
      all(2**(2*n+1) > 8*n+8 and (2*n+1)*(4*n+3) > 8*n+8 for n in range(2, 41)))
check("m(m-1)/8 is not an integer for m = 4n+3, n = 0..40, so differs from every k(m-k)",
      all(F((4*n+3)*(4*n+2), 8).denominator != 1 for n in range(41)))

# 3. Casimir of so(15) acting on 15 of the 30 directions, on the Dirac spinor of Cl(30)
X = sp.csr_matrix([[0, 1], [1, 0]]); Y = sp.csr_matrix([[0, -1j], [1j, 0]])
Z = sp.csr_matrix([[1, 0], [0, -1]]); I2 = sp.identity(2, format='csr')
def kron(ms):
    out = ms[0]
    for m in ms[1:]:
        out = sp.kron(out, m, format='csr')
    return out
def sparse_gammas(N):
    h = N // 2; g = []
    for a in range(h):
        for P in (X, Y):
            g.append(kron([Z] * a + [P] + [I2] * (h - a - 1)))
    return g
g30 = sparse_gammas(30)
C = sp.csr_matrix((2**15, 2**15), dtype=complex)
for i in range(15):
    for j in range(i + 1, 15):
        L = 0.5 * g30[i] @ g30[j]
        C = C - L @ L
check("Casimir of so(15) equals 105/4 on all 32768 spinor components of Cl(30)",
      abs(C - sp.identity(2**15) * 26.25).max() < 1e-12)

# 4. Level n = 1: compact so(7) in so(14) and so(16)
def dense_gammas(N):
    return [m.toarray() for m in sparse_gammas(N)]
def casimir_spectrum(N, gens):
    g = dense_gammas(N); dim = 2**(N // 2)
    chi = np.eye(dim, dtype=complex)
    for x in g:
        chi = chi @ x
    chi = chi * (1j) ** (-(N // 2))
    Cm = np.zeros((dim, dim), dtype=complex)
    for A in gens:
        n = A.shape[0]
        R = sum(0.25 * A[k, l] * g[k] @ g[l] for k in range(n) for l in range(n) if A[k, l] != 0)
        Cm -= R @ R
    w, v = np.linalg.eigh(chi)
    out = {}
    for sign in (1, -1):
        P = v[:, sign * w > 0]
        ev = np.round(np.linalg.eigvalsh(P.conj().T @ Cm @ P), 6) + 0.0
        vals, counts = np.unique(ev, return_counts=True)
        out[sign] = {float(a): int(b) for a, b in zip(vals, counts)}
    return out
vec7 = []
for i in range(7):
    for j in range(i + 1, 7):
        E = np.zeros((7, 7)); E[i, j] = -1; E[j, i] = 1; vec7.append(E)
# octonionic left multiplications give the spin representation of so(7) on R^8
tri = [(1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7), (5, 6, 1), (6, 7, 2), (7, 1, 3)]
M = np.zeros((8, 8, 8))
for i in range(8):
    M[0, i, i] = 1; M[i, 0, i] = 1
for i in range(1, 8):
    M[i, i, 0] = -1
for (a, b, c) in tri:
    for (x, y, z) in [(a, b, c), (b, c, a), (c, a, b)]:
        M[x, y, z] = 1; M[y, x, z] = -1
Lm = [np.array([[M[i, k, j] for k in range(8)] for j in range(8)]) for i in range(1, 8)]
check("octonion left multiplications satisfy L_iL_j + L_jL_i = -2 delta_ij and are antisymmetric",
      all(np.allclose(Lm[i] @ Lm[j] + Lm[j] @ Lm[i], -2 * np.eye(8) * (i == j)) for i in range(7) for j in range(7))
      and all(np.allclose(l, -l.T) for l in Lm))
spin7 = [0.5 * Lm[i] @ Lm[j] for i in range(7) for j in range(i + 1, 7)]
forms = sorted({k * (7 - k) for k in range(8)})
for N, half in ((14, 64), (16, 128)):
    s = casimir_spectrum(N, vec7)
    check("so(%d), so(7) on 7 directions: Casimir 21/4 on both semispinors" % N,
          s[1] == {5.25: half} and s[-1] == {5.25: half})
    s = casimir_spectrum(N, spin7)
    q = half // 16
    expect = {0.0: q, 5.25: half // 2, 6.0: 7 * q}
    check("so(%d), so(7) on 8 directions: each semispinor is %d(1+7) + %d copies of 8, no Casimir 10 or 12" % (N, q, q),
          s[1] == expect and s[-1] == expect)
check("exterior powers of R^7 have Casimir k(7-k) in {0,6,10,12}, and Lambda^3 (Casimir 12) is in both parities",
      forms == [0, 6, 10, 12] and 3 * 4 == 12 and 4 * 3 == 12)

# 5. Noncompact real forms: so(7) on 7+7+2 in so(16) gives all forms (illustrates the remark)
diag = []
for i in range(7):
    for j in range(i + 1, 7):
        E = np.zeros((14, 14)); E[i, j] = -1; E[j, i] = 1; E[7 + i, 7 + j] = -1; E[7 + j, 7 + i] = 1
        diag.append(E)
s = casimir_spectrum(16, diag)
check("so(7) on 7+7+2 directions: each semispinor of so(16) has the Casimir spectrum of all forms on R^7",
      s[1] == {0.0: 2, 6.0: 14, 10.0: 42, 12.0: 70} and s[-1] == s[1])

# 6. Signature bookkeeping for two copies of the vector
def two_copy_signatures(p, q):
    m = p + q
    return {(2 * p, 2 * q), (m, m), (2 * q, 2 * p)}
ok = True
for n in range(1, 41):
    m = 4 * n + 3
    for p in range(m + 1):
        if (8 * n + 3, 3) in two_copy_signatures(p, m - p):
            ok = False
check("two copies of R^{p,q}, p+q=4n+3, never have signature (8n+3,3), n = 1..40", ok)
ok = all(not any(s2 + r1 == 8*n+4 and t2 + (2 - r1) == 4 for (s2, t2) in two_copy_signatures(4*n+3, 0) for r1 in range(3))
         for n in range(1, 41))
check("for compact so(4n+3), two copies of R^{4n+3} plus two lines never give signature (8n+4,4)", ok)

# 7. Brane duality bookkeeping (Sec. 6)
check("11 and 15 are dual in D=30, 10 and 14 in D=28, 10 and 13 in D=27, 10 and 14 are not dual in D=27",
      11 + 15 == 30 - 4 and 10 + 14 == 28 - 4 and 10 + 13 == 27 - 4 and 10 + 14 != 27 - 4)

# 8. Weyl dimensions used in Sec. 3: dim V(2w1) = l(2l+3), dim V(w1+wl) = 2l*2^l, dim V(wl) = 2^l
from fractions import Fraction
def weyl_dim(l, lam):
    rho = [Fraction(2*(l - i) - 1, 2) for i in range(l)]
    pos = []
    for i in range(l):
        e = [0]*l; e[i] = 1; pos.append(e)
        for j in range(i + 1, l):
            a = [0]*l; a[i] = 1; a[j] = -1; pos.append(a)
            b = [0]*l; b[i] = 1; b[j] = 1; pos.append(b)
    d = Fraction(1)
    for a in pos:
        d *= sum((x + r) * c for x, r, c in zip(lam, rho, a)) / sum(r * c for r, c in zip(rho, a))
    return d
ok = True
for l in range(3, 13):
    w1 = [1] + [0]*(l - 1); wl = [Fraction(1, 2)]*l
    ok &= weyl_dim(l, [2*x for x in w1]) == l*(2*l + 3)
    ok &= weyl_dim(l, wl) == 2**l
    ok &= weyl_dim(l, [x + y for x, y in zip(w1, wl)]) == 2*l*2**l
    ok &= weyl_dim(l, [2*x for x in wl]) == __import__("math").comb(2*l + 1, l)
check("Weyl formula: dim V(2w1)=l(2l+3), V(wl)=2^l, V(w1+wl)=2l 2^l, V(2wl)=C(2l+1,l), l = 3..12", ok)
check("(2n+1)(4n+3)-(8n+8) = 8n^2+2n-5 > 0 for n = 1..40",
      all((2*n+1)*(4*n+3)-(8*n+8) == 8*n*n+2*n-5 > 0 for n in range(1, 41)))

print("%d checks passed, %d failed" % (sum(results), len(results) - sum(results)))
