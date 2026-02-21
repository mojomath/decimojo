"""Tests for BigInt2 number theory operations: GCD, Extended GCD, LCM,
modular exponentiation, and modular inverse."""

import testing
from decimojo.bigint2.bigint2 import BigInt2
from decimojo.bigint2.number_theory import (
    gcd,
    extended_gcd,
    lcm,
    mod_pow,
    mod_inverse,
)


# ===----------------------------------------------------------------------=== #
# GCD tests
# ===----------------------------------------------------------------------=== #


fn test_gcd_zero_cases() raises:
    """gcd involving zeros follows Python convention."""
    testing.assert_equal(String(gcd(BigInt2(0), BigInt2(0))), "0")
    testing.assert_equal(String(gcd(BigInt2(12), BigInt2(0))), "12")
    testing.assert_equal(String(gcd(BigInt2(0), BigInt2(15))), "15")
    testing.assert_equal(String(gcd(BigInt2(-7), BigInt2(0))), "7")
    testing.assert_equal(String(gcd(BigInt2(0), BigInt2(-9))), "9")


fn test_gcd_basic() raises:
    """Basic GCD of small positive integers."""
    testing.assert_equal(String(gcd(BigInt2(12), BigInt2(8))), "4")
    testing.assert_equal(String(gcd(BigInt2(48), BigInt2(18))), "6")
    testing.assert_equal(String(gcd(BigInt2(100), BigInt2(75))), "25")
    testing.assert_equal(String(gcd(BigInt2(17), BigInt2(13))), "1")
    testing.assert_equal(String(gcd(BigInt2(7), BigInt2(7))), "7")
    testing.assert_equal(String(gcd(BigInt2(1), BigInt2(100))), "1")
    testing.assert_equal(String(gcd(BigInt2(54), BigInt2(24))), "6")
    testing.assert_equal(String(gcd(BigInt2(270), BigInt2(192))), "6")


fn test_gcd_negative() raises:
    """GCD is always non-negative regardless of input signs."""
    testing.assert_equal(String(gcd(BigInt2(-12), BigInt2(8))), "4")
    testing.assert_equal(String(gcd(BigInt2(12), BigInt2(-8))), "4")
    testing.assert_equal(String(gcd(BigInt2(-12), BigInt2(-8))), "4")
    testing.assert_equal(String(gcd(BigInt2(-48), BigInt2(18))), "6")
    testing.assert_equal(String(gcd(BigInt2(-48), BigInt2(-18))), "6")


fn test_gcd_commutativity() raises:
    """gcd(a, b) == gcd(b, a)."""
    testing.assert_equal(
        String(gcd(BigInt2(48), BigInt2(18))),
        String(gcd(BigInt2(18), BigInt2(48))),
    )
    testing.assert_equal(
        String(gcd(BigInt2(100), BigInt2(75))),
        String(gcd(BigInt2(75), BigInt2(100))),
    )
    testing.assert_equal(
        String(gcd(BigInt2(17), BigInt2(13))),
        String(gcd(BigInt2(13), BigInt2(17))),
    )


fn test_gcd_one() raises:
    """gcd(1, n) = 1 and gcd(n, 1) = 1."""
    testing.assert_equal(String(gcd(BigInt2(1), BigInt2(999999))), "1")
    testing.assert_equal(String(gcd(BigInt2(999999), BigInt2(1))), "1")
    testing.assert_equal(String(gcd(BigInt2(1), BigInt2(1))), "1")


fn test_gcd_same_value() raises:
    """gcd(n, n) = |n|."""
    testing.assert_equal(String(gcd(BigInt2(42), BigInt2(42))), "42")
    testing.assert_equal(String(gcd(BigInt2(-42), BigInt2(42))), "42")
    testing.assert_equal(String(gcd(BigInt2(-42), BigInt2(-42))), "42")


fn test_gcd_common_factor() raises:
    """gcd(a*m, b*m) = m when gcd(a,b) = 1."""
    # gcd(17*m, 13*m) = m since gcd(17,13) = 1
    var m = BigInt2(123456789)
    var a = m * BigInt2(17)
    var b = m * BigInt2(13)
    testing.assert_equal(String(gcd(a, b)), "123456789")


fn test_gcd_powers_of_two() raises:
    """GCD of powers of two uses the binary GCD fast path."""
    # gcd(2^16, 2^8) = 2^8 = 256
    var a = BigInt2(1) << 16
    var b = BigInt2(1) << 8
    testing.assert_equal(String(gcd(a, b)), "256")

    # gcd(2^32, 2^16) = 2^16 = 65536
    var c = BigInt2(1) << 32
    var d = BigInt2(1) << 16
    testing.assert_equal(String(gcd(c, d)), "65536")

    # gcd(2^64, 2^32) = 2^32 = 4294967296
    var e = BigInt2(1) << 64
    var f = BigInt2(1) << 32
    testing.assert_equal(String(gcd(e, f)), "4294967296")


fn test_gcd_large_coprime() raises:
    """Large coprime primes."""
    var p = BigInt2(1000000007)
    var q = BigInt2(999999937)
    testing.assert_equal(String(gcd(p, q)), "1")


# ===----------------------------------------------------------------------=== #
# Extended GCD tests
# ===----------------------------------------------------------------------=== #


fn test_extended_gcd_basic() raises:
    """Extended GCD with known Bézout coefficients."""
    # gcd(35, 15) = 5, with 35*1 + 15*(-2) = 5
    var r = extended_gcd(BigInt2(35), BigInt2(15))
    testing.assert_equal(String(r[0]), "5")
    # Verify Bézout identity: a*x + b*y = gcd
    var check = BigInt2(35) * r[1] + BigInt2(15) * r[2]
    testing.assert_equal(String(check), "5")


fn test_extended_gcd_240_46() raises:
    """Extended GCD(240, 46) = 2."""
    var r = extended_gcd(BigInt2(240), BigInt2(46))
    testing.assert_equal(String(r[0]), "2")
    var check = BigInt2(240) * r[1] + BigInt2(46) * r[2]
    testing.assert_equal(String(check), "2")


fn test_extended_gcd_coprime() raises:
    """Extended GCD of coprime numbers gives gcd=1."""
    var r = extended_gcd(BigInt2(17), BigInt2(13))
    testing.assert_equal(String(r[0]), "1")
    var check = BigInt2(17) * r[1] + BigInt2(13) * r[2]
    testing.assert_equal(String(check), "1")


fn test_extended_gcd_zero_cases() raises:
    """Extended GCD with zero arguments."""
    var r1 = extended_gcd(BigInt2(0), BigInt2(5))
    testing.assert_equal(String(r1[0]), "5")
    var check1 = BigInt2(0) * r1[1] + BigInt2(5) * r1[2]
    testing.assert_equal(String(check1), "5")

    var r2 = extended_gcd(BigInt2(7), BigInt2(0))
    testing.assert_equal(String(r2[0]), "7")
    var check2 = BigInt2(7) * r2[1] + BigInt2(0) * r2[2]
    testing.assert_equal(String(check2), "7")

    var r3 = extended_gcd(BigInt2(0), BigInt2(0))
    testing.assert_equal(String(r3[0]), "0")


fn test_extended_gcd_negative() raises:
    """Extended GCD correctly adjusts signs for negative inputs."""
    var r1 = extended_gcd(BigInt2(-35), BigInt2(15))
    testing.assert_equal(String(r1[0]), "5")
    var check1 = BigInt2(-35) * r1[1] + BigInt2(15) * r1[2]
    testing.assert_equal(String(check1), "5")

    var r2 = extended_gcd(BigInt2(35), BigInt2(-15))
    testing.assert_equal(String(r2[0]), "5")
    var check2 = BigInt2(35) * r2[1] + BigInt2(-15) * r2[2]
    testing.assert_equal(String(check2), "5")

    var r3 = extended_gcd(BigInt2(-35), BigInt2(-15))
    testing.assert_equal(String(r3[0]), "5")
    var check3 = BigInt2(-35) * r3[1] + BigInt2(-15) * r3[2]
    testing.assert_equal(String(check3), "5")


fn test_extended_gcd_bezout_various() raises:
    """Verify Bézout identity for a range of input pairs."""

    # (a, b) pairs
    fn _check(a_val: Int, b_val: Int) raises:
        var a = BigInt2(a_val)
        var b = BigInt2(b_val)
        var r = extended_gcd(a, b)
        var check = BigInt2(a_val) * r[1] + BigInt2(b_val) * r[2]
        testing.assert_equal(
            String(check),
            String(r[0]),
            "Bézout identity failed for ("
            + String(a_val)
            + ", "
            + String(b_val)
            + ")",
        )

    _check(6, 4)
    _check(100, 35)
    _check(1024, 768)
    _check(999, 111)
    _check(1, 1)
    _check(2, 3)


# ===----------------------------------------------------------------------=== #
# LCM tests
# ===----------------------------------------------------------------------=== #


fn test_lcm_basic() raises:
    """Basic LCM of small integers."""
    testing.assert_equal(String(lcm(BigInt2(12), BigInt2(18))), "36")
    testing.assert_equal(String(lcm(BigInt2(4), BigInt2(6))), "12")
    testing.assert_equal(String(lcm(BigInt2(7), BigInt2(13))), "91")
    testing.assert_equal(String(lcm(BigInt2(6), BigInt2(6))), "6")
    testing.assert_equal(String(lcm(BigInt2(1), BigInt2(100))), "100")


fn test_lcm_zero() raises:
    """lcm(0, n) = lcm(n, 0) = 0."""
    testing.assert_equal(String(lcm(BigInt2(0), BigInt2(5))), "0")
    testing.assert_equal(String(lcm(BigInt2(5), BigInt2(0))), "0")
    testing.assert_equal(String(lcm(BigInt2(0), BigInt2(0))), "0")


fn test_lcm_negative() raises:
    """LCM is always non-negative."""
    testing.assert_equal(String(lcm(BigInt2(-4), BigInt2(6))), "12")
    testing.assert_equal(String(lcm(BigInt2(4), BigInt2(-6))), "12")
    testing.assert_equal(String(lcm(BigInt2(-4), BigInt2(-6))), "12")


fn test_lcm_gcd_product_identity() raises:
    """gcd(a,b) * lcm(a,b) = |a * b| for non-zero a, b."""
    var a = BigInt2(48)
    var b = BigInt2(18)
    var g = gcd(a, b)
    var l = lcm(a, b)
    testing.assert_equal(String(g * l), String(a * b))

    var c = BigInt2(100)
    var d = BigInt2(75)
    testing.assert_equal(String(gcd(c, d) * lcm(c, d)), String(c * d))


# ===----------------------------------------------------------------------=== #
# Modular exponentiation tests
# ===----------------------------------------------------------------------=== #


fn test_mod_pow_basic() raises:
    """Basic modular exponentiation."""
    # 2^10 mod 1000 = 1024 mod 1000 = 24
    testing.assert_equal(
        String(mod_pow(BigInt2(2), BigInt2(10), BigInt2(1000))), "24"
    )
    # 3^13 mod 100 = 1594323 mod 100 = 23
    testing.assert_equal(
        String(mod_pow(BigInt2(3), BigInt2(13), BigInt2(100))), "23"
    )
    # 5^1 mod 3 = 5 mod 3 = 2
    testing.assert_equal(
        String(mod_pow(BigInt2(5), BigInt2(1), BigInt2(3))), "2"
    )
    # 7^2 mod 10 = 49 mod 10 = 9
    testing.assert_equal(
        String(mod_pow(BigInt2(7), BigInt2(2), BigInt2(10))), "9"
    )


fn test_mod_pow_edge_cases() raises:
    """Edge cases: zero exponent, zero base, modulus 1."""
    # base^0 = 1 (for mod > 1)
    testing.assert_equal(
        String(mod_pow(BigInt2(5), BigInt2(0), BigInt2(3))), "1"
    )
    # 0^n = 0 for n > 0
    testing.assert_equal(
        String(mod_pow(BigInt2(0), BigInt2(10), BigInt2(7))), "0"
    )
    # x mod 1 = 0 always
    testing.assert_equal(
        String(mod_pow(BigInt2(5), BigInt2(10), BigInt2(1))), "0"
    )
    # 0^0 mod m = 1 for m > 1 (convention: 0^0 = 1)
    testing.assert_equal(
        String(mod_pow(BigInt2(0), BigInt2(0), BigInt2(5))), "1"
    )
    # Any^Any mod 1 = 0
    testing.assert_equal(
        String(mod_pow(BigInt2(0), BigInt2(0), BigInt2(1))), "0"
    )


fn test_mod_pow_fermat_little_theorem() raises:
    """Fermat: a^(p-1) ≡ 1 (mod p) for prime p, gcd(a,p)=1."""
    # 3^6 mod 7 = 1
    testing.assert_equal(
        String(mod_pow(BigInt2(3), BigInt2(6), BigInt2(7))), "1"
    )
    # 5^12 mod 13 = 1
    testing.assert_equal(
        String(mod_pow(BigInt2(5), BigInt2(12), BigInt2(13))), "1"
    )
    # 2^16 mod 17 = 1
    testing.assert_equal(
        String(mod_pow(BigInt2(2), BigInt2(16), BigInt2(17))), "1"
    )


fn test_mod_pow_larger_exponent() raises:
    """Larger exponents that benefit from binary exponentiation."""
    # 7^256 mod 13 = 9
    testing.assert_equal(
        String(mod_pow(BigInt2(7), BigInt2(256), BigInt2(13))), "9"
    )
    # 2^32 mod 1000000007 = 294967268
    testing.assert_equal(
        String(mod_pow(BigInt2(2), BigInt2(32), BigInt2(1000000007))),
        "294967268",
    )


fn test_mod_pow_negative_base() raises:
    """Negative base is reduced via floor modulo first."""
    # (-2)^3 mod 5 = (-8) mod 5 = 2
    testing.assert_equal(
        String(mod_pow(BigInt2(-2), BigInt2(3), BigInt2(5))), "2"
    )
    # (-3)^2 mod 7 = 9 mod 7 = 2
    testing.assert_equal(
        String(mod_pow(BigInt2(-3), BigInt2(2), BigInt2(7))), "2"
    )
    # (-1)^1000 mod 7 = 1
    testing.assert_equal(
        String(mod_pow(BigInt2(-1), BigInt2(1000), BigInt2(7))), "1"
    )
    # (-1)^999 mod 7 = 6 (since -1 mod 7 = 6, 6^odd mod 7 = 6)
    testing.assert_equal(
        String(mod_pow(BigInt2(-1), BigInt2(999), BigInt2(7))), "6"
    )


fn test_mod_pow_int_overload() raises:
    """Convenience overload with Int exponent."""
    testing.assert_equal(String(mod_pow(BigInt2(2), 10, BigInt2(1000))), "24")
    testing.assert_equal(String(mod_pow(BigInt2(3), 13, BigInt2(100))), "23")


fn test_mod_pow_errors() raises:
    """mod_pow raises on negative exponent or non-positive modulus."""
    var raised = False

    # Negative exponent
    try:
        var _ = mod_pow(BigInt2(2), BigInt2(-1), BigInt2(5))
    except:
        raised = True
    testing.assert_true(raised, "Should raise on negative exponent")

    # Zero modulus
    raised = False
    try:
        var _ = mod_pow(BigInt2(2), BigInt2(3), BigInt2(0))
    except:
        raised = True
    testing.assert_true(raised, "Should raise on zero modulus")

    # Negative modulus
    raised = False
    try:
        var _ = mod_pow(BigInt2(2), BigInt2(3), BigInt2(-5))
    except:
        raised = True
    testing.assert_true(raised, "Should raise on negative modulus")


# ===----------------------------------------------------------------------=== #
# Modular inverse tests
# ===----------------------------------------------------------------------=== #


fn test_mod_inverse_basic() raises:
    """Known modular inverses."""
    # 3 * 5 = 15 ≡ 1 (mod 7)
    testing.assert_equal(String(mod_inverse(BigInt2(3), BigInt2(7))), "5")
    # 7 * 8 = 56 ≡ 1 (mod 11)
    testing.assert_equal(String(mod_inverse(BigInt2(7), BigInt2(11))), "8")
    # 3 * 4 = 12 ≡ 1 (mod 11)
    testing.assert_equal(String(mod_inverse(BigInt2(3), BigInt2(11))), "4")


fn test_mod_inverse_one() raises:
    """Inverse of 1 is always 1."""
    testing.assert_equal(String(mod_inverse(BigInt2(1), BigInt2(7))), "1")
    testing.assert_equal(String(mod_inverse(BigInt2(1), BigInt2(100))), "1")


fn test_mod_inverse_verify_roundtrip() raises:
    """Verify (a * inv(a)) mod m == 1."""
    var a = BigInt2(17)
    var m = BigInt2(100)
    var inv = mod_inverse(a, m)
    testing.assert_equal(String((a * inv) % m), "1")

    var a2 = BigInt2(37)
    var m2 = BigInt2(1000)
    var inv2 = mod_inverse(a2, m2)
    testing.assert_equal(String((a2 * inv2) % m2), "1")


fn test_mod_inverse_negative_input() raises:
    """Modular inverse of a negative number."""
    # (-3) mod 7 = 4, inv(4) mod 7: 4*2=8≡1 mod 7 → inv=2
    var inv = mod_inverse(BigInt2(-3), BigInt2(7))
    testing.assert_equal(String((BigInt2(-3) * inv) % BigInt2(7)), "1")


fn test_mod_inverse_not_exists() raises:
    """mod_inverse raises when gcd(a, m) != 1."""
    var raised = False
    try:
        var _ = mod_inverse(BigInt2(2), BigInt2(6))
    except:
        raised = True
    testing.assert_true(raised, "mod_inverse(2, 6) should raise")

    raised = False
    try:
        var _ = mod_inverse(BigInt2(4), BigInt2(8))
    except:
        raised = True
    testing.assert_true(raised, "mod_inverse(4, 8) should raise")

    raised = False
    try:
        var _ = mod_inverse(BigInt2(0), BigInt2(5))
    except:
        raised = True
    testing.assert_true(raised, "mod_inverse(0, 5) should raise")


# ===----------------------------------------------------------------------=== #
# Main
# ===----------------------------------------------------------------------=== #


fn main() raises:
    # GCD tests
    test_gcd_zero_cases()
    print("PASS: test_gcd_zero_cases")
    test_gcd_basic()
    print("PASS: test_gcd_basic")
    test_gcd_negative()
    print("PASS: test_gcd_negative")
    test_gcd_commutativity()
    print("PASS: test_gcd_commutativity")
    test_gcd_one()
    print("PASS: test_gcd_one")
    test_gcd_same_value()
    print("PASS: test_gcd_same_value")
    test_gcd_common_factor()
    print("PASS: test_gcd_common_factor")
    test_gcd_powers_of_two()
    print("PASS: test_gcd_powers_of_two")
    test_gcd_large_coprime()
    print("PASS: test_gcd_large_coprime")

    # Extended GCD tests
    test_extended_gcd_basic()
    print("PASS: test_extended_gcd_basic")
    test_extended_gcd_240_46()
    print("PASS: test_extended_gcd_240_46")
    test_extended_gcd_coprime()
    print("PASS: test_extended_gcd_coprime")
    test_extended_gcd_zero_cases()
    print("PASS: test_extended_gcd_zero_cases")
    test_extended_gcd_negative()
    print("PASS: test_extended_gcd_negative")
    test_extended_gcd_bezout_various()
    print("PASS: test_extended_gcd_bezout_various")

    # LCM tests
    test_lcm_basic()
    print("PASS: test_lcm_basic")
    test_lcm_zero()
    print("PASS: test_lcm_zero")
    test_lcm_negative()
    print("PASS: test_lcm_negative")
    test_lcm_gcd_product_identity()
    print("PASS: test_lcm_gcd_product_identity")

    # mod_pow tests
    test_mod_pow_basic()
    print("PASS: test_mod_pow_basic")
    test_mod_pow_edge_cases()
    print("PASS: test_mod_pow_edge_cases")
    test_mod_pow_fermat_little_theorem()
    print("PASS: test_mod_pow_fermat_little_theorem")
    test_mod_pow_larger_exponent()
    print("PASS: test_mod_pow_larger_exponent")
    test_mod_pow_negative_base()
    print("PASS: test_mod_pow_negative_base")
    test_mod_pow_int_overload()
    print("PASS: test_mod_pow_int_overload")
    test_mod_pow_errors()
    print("PASS: test_mod_pow_errors")

    # mod_inverse tests
    test_mod_inverse_basic()
    print("PASS: test_mod_inverse_basic")
    test_mod_inverse_one()
    print("PASS: test_mod_inverse_one")
    test_mod_inverse_verify_roundtrip()
    print("PASS: test_mod_inverse_verify_roundtrip")
    test_mod_inverse_negative_input()
    print("PASS: test_mod_inverse_negative_input")
    test_mod_inverse_not_exists()
    print("PASS: test_mod_inverse_not_exists")

    print("\n=== All 31 number theory tests passed ===")
