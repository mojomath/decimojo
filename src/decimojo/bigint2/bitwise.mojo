"""Bitwise operations for BigInt2: AND, OR, XOR, NOT.

All operations follow Python's semantics for arbitrary-precision integers,
treating negative numbers as having an infinite-width two's complement
representation. That is:

    -x  is conceptually represented as  ...111111 (flip all bits of x-1)

This means:
    ~x  = -(x + 1)         for all x
    -1  = ...11111111       (all bits set)
    -2  = ...11111110
    -3  = ...11111101

For binary operations (AND, OR, XOR), the algorithm is:
1. Convert negative operands to two's complement form:
   negate magnitude: subtract 1, then invert all words.
   The "sign extension" is implicitly all-1s for negative numbers.
2. Perform word-by-word bitwise operation, extending shorter operand
   with its sign-extension fill (0x00000000 for non-negative, 0xFFFFFFFF
   for negative).
3. Determine the result sign from the operation on the sign-extension bits.
4. If the result is negative (in two's complement), convert back to
   sign-magnitude: invert all words, then add 1 to the magnitude.

Performance note: For the common case of two non-negative operands, no
two's complement conversion is needed — just word-by-word operation.
"""

from decimojo.bigint2.bigint2 import BigInt2


# ===----------------------------------------------------------------------=== #
# Core helper: word-by-word binary bitwise operation
# ===----------------------------------------------------------------------=== #


fn _binary_bitwise_op[op: StringLiteral](a: BigInt2, b: BigInt2) -> BigInt2:
    """Performs a word-by-word binary bitwise operation on two BigInt2 values.

    The operation is determined by the `op` parameter:
    - "and": bitwise AND
    - "or":  bitwise OR
    - "xor": bitwise XOR

    Uses Python-compatible two's complement semantics for negative numbers.
    """

    # Determine fills (sign extension for infinite-width two's complement)
    var a_fill = UInt32(0xFFFF_FFFF) if a.sign else UInt32(0)
    var b_fill = UInt32(0xFFFF_FFFF) if b.sign else UInt32(0)

    # Determine result sign from operation on sign-extension bits
    var result_negative: Bool

    @parameter
    if op == "and":
        result_negative = a.sign and b.sign
    elif op == "or":
        result_negative = a.sign or b.sign
    elif op == "xor":
        result_negative = a.sign != b.sign
    else:
        constrained[False, "op must be 'and', 'or', or 'xor'"]()
        result_negative = False  # unreachable

    # Fast path: both non-negative
    if not a.sign and not b.sign:

        @parameter
        if op == "and":
            # AND with zeros → zeros, so result is at most min_len words
            var min_len = min(len(a.words), len(b.words))
            var result_words = List[UInt32](capacity=min_len)
            for i in range(min_len):
                result_words.append(a.words[i] & b.words[i])
            while len(result_words) > 1 and result_words[-1] == 0:
                result_words.shrink(len(result_words) - 1)
            return BigInt2(raw_words=result_words^, sign=False)
        elif op == "or":
            var max_len = max(len(a.words), len(b.words))
            var result_words = List[UInt32](capacity=max_len)
            for i in range(max_len):
                var wa = UInt32(0) if i >= len(a.words) else a.words[i]
                var wb = UInt32(0) if i >= len(b.words) else b.words[i]
                result_words.append(wa | wb)
            while len(result_words) > 1 and result_words[-1] == 0:
                result_words.shrink(len(result_words) - 1)
            return BigInt2(raw_words=result_words^, sign=False)
        else:  # xor
            var max_len = max(len(a.words), len(b.words))
            var result_words = List[UInt32](capacity=max_len)
            for i in range(max_len):
                var wa = UInt32(0) if i >= len(a.words) else a.words[i]
                var wb = UInt32(0) if i >= len(b.words) else b.words[i]
                result_words.append(wa ^ wb)
            while len(result_words) > 1 and result_words[-1] == 0:
                result_words.shrink(len(result_words) - 1)
            return BigInt2(raw_words=result_words^, sign=False)

    # General path: convert negative operands to two's complement inline
    var a_n = len(a.words)
    var b_n = len(b.words)
    var max_len = max(a_n, b_n)
    # Negative results may need an extra word for the conversion back
    if result_negative:
        max_len += 1

    # Pre-compute a's TC words: for negative, TC = ~(|a| - 1)
    var a_tc = List[UInt32](capacity=a_n)
    if a.sign:
        var borrow: UInt64 = 1
        for i in range(a_n):
            var diff = UInt64(a.words[i]) - borrow
            a_tc.append(~UInt32(diff & 0xFFFF_FFFF))
            borrow = (diff >> 63) & 1
    else:
        for i in range(a_n):
            a_tc.append(a.words[i])

    # Pre-compute b's TC words
    var b_tc = List[UInt32](capacity=b_n)
    if b.sign:
        var borrow: UInt64 = 1
        for i in range(b_n):
            var diff = UInt64(b.words[i]) - borrow
            b_tc.append(~UInt32(diff & 0xFFFF_FFFF))
            borrow = (diff >> 63) & 1
    else:
        for i in range(b_n):
            b_tc.append(b.words[i])

    # Perform the operation word-by-word
    var result_tc = List[UInt32](capacity=max_len)
    for i in range(max_len):
        var wa = a_fill if i >= len(a_tc) else a_tc[i]
        var wb = b_fill if i >= len(b_tc) else b_tc[i]

        @parameter
        if op == "and":
            result_tc.append(wa & wb)
        elif op == "or":
            result_tc.append(wa | wb)
        else:  # xor
            result_tc.append(wa ^ wb)

    # Convert result back from two's complement if negative
    if not result_negative:
        # Non-negative: result_tc is the magnitude
        while len(result_tc) > 1 and result_tc[-1] == 0:
            result_tc.shrink(len(result_tc) - 1)
        if len(result_tc) == 1 and result_tc[0] == 0:
            return BigInt2()
        return BigInt2(raw_words=result_tc^, sign=False)
    else:
        # Negative: magnitude = ~result_tc + 1
        var n = len(result_tc)
        var mag = List[UInt32](capacity=n)
        for i in range(n):
            mag.append(~result_tc[i])
        var carry: UInt64 = 1
        for i in range(len(mag)):
            var s = UInt64(mag[i]) + carry
            mag[i] = UInt32(s & 0xFFFF_FFFF)
            carry = s >> 32
            if carry == 0:
                break
        if carry > 0:
            mag.append(UInt32(carry))
        # Strip leading zeros
        while len(mag) > 1 and mag[-1] == 0:
            mag.shrink(len(mag) - 1)
        if len(mag) == 1 and mag[0] == 0:
            return BigInt2()
        return BigInt2(raw_words=mag^, sign=True)


# ===----------------------------------------------------------------------=== #
# Core helper: in-place word-by-word binary bitwise operation
# ===----------------------------------------------------------------------=== #


fn _binary_bitwise_op_inplace[
    op: StringLiteral
](mut a: BigInt2, read b: BigInt2):
    """Performs a word-by-word binary bitwise operation on `a` in-place.

    Computes the result word list and moves it into a.words, avoiding
    full BigInt2 construction.

    The operation is determined by the `op` parameter:
    - "and": bitwise AND
    - "or":  bitwise OR
    - "xor": bitwise XOR
    """

    # Determine fills (sign extension for infinite-width two's complement)
    var a_fill = UInt32(0xFFFF_FFFF) if a.sign else UInt32(0)
    var b_fill = UInt32(0xFFFF_FFFF) if b.sign else UInt32(0)

    # Determine result sign from operation on sign-extension bits
    var result_negative: Bool

    @parameter
    if op == "and":
        result_negative = a.sign and b.sign
    elif op == "or":
        result_negative = a.sign or b.sign
    elif op == "xor":
        result_negative = a.sign != b.sign
    else:
        constrained[False, "op must be 'and', 'or', or 'xor'"]()
        result_negative = False  # unreachable

    # Fast path: both non-negative
    if not a.sign and not b.sign:

        @parameter
        if op == "and":
            var min_len = min(len(a.words), len(b.words))
            # We can modify a.words in-place for AND (result <= min_len)
            for i in range(min_len):
                a.words[i] = a.words[i] & b.words[i]
            # Truncate to min_len in a single shrink call
            if len(a.words) > min_len:
                a.words.shrink(min_len)
            # Strip leading zeros
            while len(a.words) > 1 and a.words[-1] == 0:
                a.words.shrink(len(a.words) - 1)
            return
        elif op == "or":
            var b_len = len(b.words)
            # Extend a if b is longer
            while len(a.words) < b_len:
                a.words.append(UInt32(0))
            for i in range(b_len):
                a.words[i] = a.words[i] | b.words[i]
            # Words beyond b_len remain as-is (OR with 0)
            while len(a.words) > 1 and a.words[-1] == 0:
                a.words.shrink(len(a.words) - 1)
            return
        else:  # xor
            var b_len = len(b.words)
            while len(a.words) < b_len:
                a.words.append(UInt32(0))
            for i in range(b_len):
                a.words[i] = a.words[i] ^ b.words[i]
            # Words beyond b_len remain as-is (XOR with 0)
            while len(a.words) > 1 and a.words[-1] == 0:
                a.words.shrink(len(a.words) - 1)
            return

    # General path: convert negative operands to two's complement
    var a_n = len(a.words)
    var b_n = len(b.words)
    var max_len = max(a_n, b_n)
    if result_negative:
        max_len += 1

    # Pre-compute a's TC words
    var a_tc = List[UInt32](capacity=a_n)
    if a.sign:
        var borrow: UInt64 = 1
        for i in range(a_n):
            var diff = UInt64(a.words[i]) - borrow
            a_tc.append(~UInt32(diff & 0xFFFF_FFFF))
            borrow = (diff >> 63) & 1
    else:
        for i in range(a_n):
            a_tc.append(a.words[i])

    # Pre-compute b's TC words
    var b_tc = List[UInt32](capacity=b_n)
    if b.sign:
        var borrow: UInt64 = 1
        for i in range(b_n):
            var diff = UInt64(b.words[i]) - borrow
            b_tc.append(~UInt32(diff & 0xFFFF_FFFF))
            borrow = (diff >> 63) & 1
    else:
        for i in range(b_n):
            b_tc.append(b.words[i])

    # Perform the operation word-by-word into a new list
    var result_tc = List[UInt32](capacity=max_len)
    for i in range(max_len):
        var wa = a_fill if i >= len(a_tc) else a_tc[i]
        var wb = b_fill if i >= len(b_tc) else b_tc[i]

        @parameter
        if op == "and":
            result_tc.append(wa & wb)
        elif op == "or":
            result_tc.append(wa | wb)
        else:  # xor
            result_tc.append(wa ^ wb)

    # Convert result back from two's complement if negative
    if not result_negative:
        while len(result_tc) > 1 and result_tc[-1] == 0:
            result_tc.shrink(len(result_tc) - 1)
        if len(result_tc) == 1 and result_tc[0] == 0:
            a.words.clear()
            a.words.append(UInt32(0))
            a.sign = False
        else:
            a.words = result_tc^
            a.sign = False
    else:
        var n = len(result_tc)
        var mag = List[UInt32](capacity=n)
        for i in range(n):
            mag.append(~result_tc[i])
        var carry: UInt64 = 1
        for i in range(len(mag)):
            var s = UInt64(mag[i]) + carry
            mag[i] = UInt32(s & 0xFFFF_FFFF)
            carry = s >> 32
            if carry == 0:
                break
        if carry > 0:
            mag.append(UInt32(carry))
        while len(mag) > 1 and mag[-1] == 0:
            mag.shrink(len(mag) - 1)
        if len(mag) == 1 and mag[0] == 0:
            a.words.clear()
            a.words.append(UInt32(0))
            a.sign = False
        else:
            a.words = mag^
            a.sign = True


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


fn bitwise_and(a: BigInt2, b: BigInt2) -> BigInt2:
    """Returns a & b using Python-compatible two's complement semantics."""
    return _binary_bitwise_op["and"](a, b)


fn bitwise_or(a: BigInt2, b: BigInt2) -> BigInt2:
    """Returns a | b using Python-compatible two's complement semantics."""
    return _binary_bitwise_op["or"](a, b)


fn bitwise_xor(a: BigInt2, b: BigInt2) -> BigInt2:
    """Returns a ^ b using Python-compatible two's complement semantics."""
    return _binary_bitwise_op["xor"](a, b)


fn bitwise_not(x: BigInt2) -> BigInt2:
    """Returns ~x using Python-compatible two's complement semantics.

    ~x = -(x + 1)

    For non-negative x: result is -(x+1), always negative (except ~(-1) = 0).
    For negative x (x = -|x|): result is |x| - 1, always non-negative.
    """
    if not x.sign:
        # ~non_negative = -(x + 1)
        var n = len(x.words)
        var result_words = List[UInt32](capacity=n + 1)
        var carry: UInt64 = 1
        for i in range(n):
            var s = UInt64(x.words[i]) + carry
            result_words.append(UInt32(s & 0xFFFF_FFFF))
            carry = s >> 32
        if carry > 0:
            result_words.append(UInt32(carry))
        return BigInt2(raw_words=result_words^, sign=True)
    else:
        # ~negative = |x| - 1
        var n = len(x.words)
        var result_words = List[UInt32](capacity=n)
        var borrow: UInt64 = 1
        for i in range(n):
            var diff = UInt64(x.words[i]) - borrow
            result_words.append(UInt32(diff & 0xFFFF_FFFF))
            borrow = (diff >> 63) & 1
        # Strip leading zeros
        while len(result_words) > 1 and result_words[-1] == 0:
            result_words.shrink(len(result_words) - 1)
        if len(result_words) == 1 and result_words[0] == 0:
            return BigInt2()
        return BigInt2(raw_words=result_words^, sign=False)


# ===----------------------------------------------------------------------=== #
# Public in-place API
# ===----------------------------------------------------------------------=== #


fn bitwise_and_inplace(mut a: BigInt2, read b: BigInt2):
    """Performs a &= b in-place using Python-compatible two's complement
    semantics."""
    _binary_bitwise_op_inplace["and"](a, b)


fn bitwise_or_inplace(mut a: BigInt2, read b: BigInt2):
    """Performs a |= b in-place using Python-compatible two's complement
    semantics."""
    _binary_bitwise_op_inplace["or"](a, b)


fn bitwise_xor_inplace(mut a: BigInt2, read b: BigInt2):
    """Performs a ^= b in-place using Python-compatible two's complement
    semantics."""
    _binary_bitwise_op_inplace["xor"](a, b)
