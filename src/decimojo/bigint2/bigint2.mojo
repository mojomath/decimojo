# ===----------------------------------------------------------------------=== #
# Copyright 2025 Yuhao Zhu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Implements basic object methods for the BigInt2 type.

This module contains the basic object methods for the BigInt2 type.
These methods include constructors, life time methods, output dunders,
type-transfer dunders, basic arithmetic operation dunders, comparison
operation dunders, and other dunders that implement traits, as well as
mathematical methods that do not implement a trait.

BigInt2 is the core binary-represented arbitrary-precision signed integer
for the DeciMojo library. It uses base-2^32 representation with UInt32 words
in little-endian order, and a separate sign bit.

Once BigInt2 is stable and performant, the alias `BInt` will be
reassigned from BigInt10 to BigInt2, making BigInt2 the primary integer type.
"""

from memory import UnsafePointer, memcpy

import decimojo.bigint2.arithmetics
import decimojo.bigint2.bitwise
import decimojo.bigint2.comparison
import decimojo.bigint2.exponential
import decimojo.bigint2.number_theory
import decimojo.str
from decimojo.bigint10.bigint10 import BigInt10
from decimojo.biguint.biguint import BigUInt
from decimojo.errors import DeciMojoError, ConversionError

# Type aliases
comptime BInt2 = BigInt2
"""A shorter comptime for BigInt2."""


struct BigInt2(
    Absable,
    Comparable,
    Copyable,
    IntableRaising,
    Movable,
    Representable,
    Stringable,
    Writable,
):
    """Represents an arbitrary-precision binary signed integer.

    Notes:

    Internal Representation:

    Uses base-2^32 representation for the integer magnitude.
    BigInt2 uses a dynamic structure in memory, which contains:
    - A List[UInt32] of words for the magnitude stored in little-endian order.
      Each UInt32 word uses the full 32-bit range [0, 2^32 - 1].
    - A Bool for the sign (True = negative, False = non-negative).

    The absolute value is calculated as:

    |x| = words[0] + words[1] * 2^32 + words[2] * 2^64 + ... + words[n] * 2^(32n)

    The actual value is: (-1)^sign * |x|.

    This is analogous to GMP and most modern bigint libraries that use
    native-word-sized limbs with a separate sign.

    Arithmetic intermediate results use UInt64 for single products
    (UInt32 * UInt32 → UInt64) and UInt128 for accumulation, which allows
    efficient schoolbook and Karatsuba multiplication on 64-bit hardware.
    """

    var words: List[UInt32]
    """A list of UInt32 words representing the magnitude in little-endian order.
    Each word uses the full [0, 2^32 - 1] range."""

    var sign: Bool
    """True if the number is negative, False if zero or positive."""

    # ===------------------------------------------------------------------=== #
    # Constants
    # ===------------------------------------------------------------------=== #

    comptime BITS_PER_WORD = 32
    """Number of bits per word."""
    comptime BASE: UInt64 = 1 << 32  # 4294967296
    """The base used for the BigInt2 representation (2^32)."""
    comptime WORD_MAX: UInt32 = ~UInt32(0)  # 0xFFFF_FFFF = 4294967295
    """The maximum value of a single word (2^32 - 1)."""
    comptime WORD_MASK: UInt64 = (1 << 32) - 1
    """Mask to extract the lower 32 bits from a UInt64."""

    comptime ZERO = Self.zero()
    """The value 0."""
    comptime ONE = Self.one()
    """The value 1."""

    @always_inline
    @staticmethod
    fn zero() -> Self:
        """Returns a BigInt2 with value 0."""
        return Self()

    @always_inline
    @staticmethod
    fn one() -> Self:
        """Returns a BigInt2 with value 1."""
        return Self(raw_words=[UInt32(1)], sign=False)

    @always_inline
    @staticmethod
    fn negative_one() -> Self:
        """Returns a BigInt2 with value -1."""
        return Self(raw_words=[UInt32(1)], sign=True)

    # ===------------------------------------------------------------------=== #
    # Constructors and life time dunder methods
    # ===------------------------------------------------------------------=== #

    fn __init__(out self):
        """Initializes a BigInt2 with value 0."""
        self.words = [UInt32(0)]
        self.sign = False

    fn __init__(out self, *, uninitialized_capacity: Int):
        """Creates an uninitialized BigInt2 with a given word capacity.
        The words list is empty; caller must append words before use.

        Args:
            uninitialized_capacity: The initial capacity for the words list.
        """
        self.words = List[UInt32](capacity=uninitialized_capacity)
        self.sign = False

    fn __init__(out self, *, var raw_words: List[UInt32], sign: Bool):
        """Initializes a BigInt2 from a list of raw words without
        validation. The caller must ensure words are in valid little-endian
        form with no unnecessary leading zeros.

        Args:
            raw_words: A list of UInt32 words in little-endian order.
            sign: True if negative, False if non-negative.

        Notes:
            **UNSAFE**: Does not strip leading zeros or check for -0.
            Always ensures at least one word exists.
        """
        if len(raw_words) == 0:
            self.words = [UInt32(0)]
            self.sign = False
        else:
            self.words = raw_words^
            self.sign = sign

    @implicit
    fn __init__(out self, value: Int):
        """Initializes a BigInt2 from an Int.

        Args:
            value: The integer value.
        """
        self = Self.from_int(value)

    fn __init__(out self, value: String) raises:
        """Initializes a BigInt2 from a decimal string representation.

        Args:
            value: The string representation of the integer.
        """
        self = Self.from_string(value)

    @implicit
    fn __init__(out self, value: Scalar):
        """Constructs a BigInt2 from an integral scalar.
        This includes all SIMD integral types, such as Int8, Int16, UInt32, etc.

        Constraints:
            The dtype of the scalar must be integral.
        """
        self = Self.from_integral_scalar(value)

    # ===------------------------------------------------------------------=== #
    # Constructing methods that are not dunders
    # ===------------------------------------------------------------------=== #

    @staticmethod
    fn from_int(value: Int) -> Self:
        """Creates a BigInt2 from a Mojo Int.

        Args:
            value: The integer value.

        Returns:
            The BigInt2 representation.
        """
        if value == 0:
            return Self()

        var sign: Bool
        var magnitude: UInt

        if value < 0:
            sign = True
            # Handle Int.MIN (two's complement asymmetry)
            if value == Int.MIN:
                # |Int.MIN| = Int.MAX + 1
                magnitude = UInt(Int.MAX) + 1
            else:
                magnitude = UInt(-value)
        else:
            sign = False
            magnitude = UInt(value)

        # Split the magnitude into 32-bit words
        # On 64-bit platforms, Int is 64 bits → at most 2 words
        var words = List[UInt32](capacity=2)
        while magnitude != 0:
            words.append(UInt32(magnitude & 0xFFFF_FFFF))
            magnitude >>= 32

        return Self(raw_words=words^, sign=sign)

    @staticmethod
    fn from_uint64(value: UInt64) -> Self:
        """Creates a BigInt2 from a UInt64.

        Args:
            value: The unsigned 64-bit integer value.

        Returns:
            The BigInt2 representation.
        """
        if value == 0:
            return Self()

        var words = List[UInt32](capacity=2)
        var lo = UInt32(value & 0xFFFF_FFFF)
        var hi = UInt32(value >> 32)
        words.append(lo)
        if hi != 0:
            words.append(hi)

        return Self(raw_words=words^, sign=False)

    @staticmethod
    fn from_uint128(value: UInt128) -> Self:
        """Creates a BigInt2 from a UInt128.

        Args:
            value: The unsigned 128-bit integer value.

        Returns:
            The BigInt2 representation.
        """
        if value == 0:
            return Self()

        var words = List[UInt32](capacity=4)
        var remaining = value
        while remaining != 0:
            words.append(UInt32(remaining & 0xFFFF_FFFF))
            remaining >>= 32

        return Self(raw_words=words^, sign=False)

    @staticmethod
    fn from_integral_scalar[dtype: DType, //](value: SIMD[dtype, 1]) -> Self:
        """Initializes a BigInt2 from an integral scalar.
        This includes all SIMD integral types, such as Int8, Int16, UInt32, etc.

        Constraints:
            The dtype must be integral.

        Args:
            value: The Scalar value to be converted to BigInt2.

        Returns:
            The BigInt2 representation of the Scalar value.
        """

        constrained[dtype.is_integral(), "dtype must be integral."]()

        if value == 0:
            return Self()

        var sign: Bool
        var magnitude: UInt64

        @parameter
        if dtype.is_unsigned():
            sign = False
            magnitude = UInt64(value)
        else:
            if value < 0:
                sign = True
                # Compute magnitude using explicit two's-complement conversion
                magnitude = UInt64(0) - UInt64(value)
            else:
                sign = False
                magnitude = UInt64(value)

        var words = List[UInt32](capacity=2)
        var lo = UInt32(magnitude & 0xFFFF_FFFF)
        var hi = UInt32(magnitude >> 32)
        words.append(lo)
        if hi != 0:
            words.append(hi)

        return Self(raw_words=words^, sign=sign)

    @staticmethod
    fn from_string(value: String) raises -> Self:
        """Creates a BigInt2 from a string representation.
        The string is normalized with `decimojo.str.parse_numeric_string()`.

        Supports signs, commas, underscores, spaces, scientific notation,
        and decimal points (the fractional part must be zero for integers).

        Uses divide-and-conquer base conversion for large numbers
        (O(M(n)·log n)) and simple multiply-and-add for small numbers (O(n²)).

        Args:
            value: The string representation (e.g. "12345", "-98765",
                "1_000_000", "1.23e5", "1,234,567").

        Returns:
            The BigInt2 representation.

        Raises:
            Error: If the string is empty, contains invalid characters,
                or represents a non-integer value.
        """
        # Use the shared string parser for format handling
        _tuple = decimojo.str.parse_numeric_string(value)
        var ref coef: List[UInt8] = _tuple[0]
        var scale: Int = _tuple[1]
        var sign: Bool = _tuple[2]

        # Check if the number is zero
        if len(coef) == 1 and coef[0] == UInt8(0):
            return Self()

        # Handle scale: positive scale means fractional digits exist.
        # For BigInt2 (integer type), the fractional part must be zero.
        if scale > 0:
            if scale >= len(coef):
                raise Error(
                    ConversionError(
                        file="src/decimojo/bigint2/bigint2.mojo",
                        function="BigInt2.from_string(value: String)",
                        message=(
                            'The input value "'
                            + value
                            + '" is not an integer.\n'
                            + "The scale is larger than the number of digits."
                        ),
                        previous_error=None,
                    )
                )
            # Check that the fractional digits are all zero
            for i in range(1, scale + 1):
                if coef[-i] != 0:
                    raise Error(
                        ConversionError(
                            file="src/decimojo/bigint2/bigint2.mojo",
                            function="BigInt2.from_string(value: String)",
                            message=(
                                'The input value "'
                                + value
                                + '" is not an integer.\n'
                                + "The fractional part is not zero."
                            ),
                            previous_error=None,
                        )
                    )
            # Remove fractional zeros from coefficient
            coef.resize(len(coef) - scale, UInt8(0))

        # Handle negative scale: it means trailing zeros to append.
        # e.g. "1.234e8" -> coef=[1,2,3,4], scale=-4, meaning 12340000
        if scale < 0:
            var zeros_to_add = -scale
            for _ in range(zeros_to_add):
                coef.append(0)

        var digit_count = len(coef)

        # coef already contains digit values 0-9, pass directly.
        # Choose conversion strategy based on digit count.
        var result: Self
        if digit_count <= _DC_FROM_STR_ENTRY_THRESHOLD:
            result = _from_decimal_digits_simple(coef, 0, digit_count)
        else:
            try:
                result = _from_decimal_digits_dc(coef, 0, digit_count)
            except e:
                # Fallback to simple O(n²) method if D&C raises an Error
                result = _from_decimal_digits_simple(coef, 0, digit_count)

        result.sign = sign
        return result^

    @staticmethod
    fn from_bigint10(value: BigInt10) -> Self:
        """Converts a base-10^9 BigInt10 to a base-2^32 BigInt2.

        Args:
            value: The BigInt10 (base-10^9) to convert.

        Returns:
            The BigInt2 (base-2^32) representation.
        """
        if value.is_zero():
            return Self()

        # Convert from base 10^9 to base 2^32 using repeated division
        # Work on the magnitude words (base-10^9)
        var div_words = List[UInt32](capacity=len(value.magnitude.words))
        for word in value.magnitude.words:
            div_words.append(word)
        var result = Self(uninitialized_capacity=len(value.magnitude.words))

        var all_zero = False
        while not all_zero:
            var remainder: UInt64 = 0
            for i in range(len(div_words) - 1, -1, -1):
                var temp = remainder * UInt64(BigUInt.BASE) + UInt64(
                    div_words[i]
                )
                div_words[i] = UInt32(temp >> 32)
                remainder = temp & 0xFFFF_FFFF

            # Remove leading zeros from dividend
            while len(div_words) > 1 and div_words[-1] == 0:
                div_words.shrink(len(div_words) - 1)

            result.words.append(UInt32(remainder))

            # Check if dividend is zero
            all_zero = True
            for word in div_words:
                if word != 0:
                    all_zero = False
                    break

        result.sign = value.sign
        return result^

    # ===------------------------------------------------------------------=== #
    # Output dunders, type-transfer dunders
    # ===------------------------------------------------------------------=== #

    fn __int__(self) raises -> Int:
        """Returns the number as Int.
        See `to_int()` for more information.
        """
        return self.to_int()

    fn __str__(self) -> String:
        """Returns a decimal string representation of the BigInt2."""
        return self.to_decimal_string()

    fn __repr__(self) -> String:
        """Returns a debug representation of the BigInt2."""
        return 'BigInt2("' + self.to_decimal_string() + '")'

    fn write_to[W: Writer](self, mut writer: W):
        """Writes the decimal string representation to a writer."""
        writer.write(self.to_decimal_string())

    # ===------------------------------------------------------------------=== #
    # Type-transfer or output methods that are not dunders
    # ===------------------------------------------------------------------=== #

    fn to_int(self) raises -> Int:
        """Returns the number as Int.

        Returns:
            The number as Int.

        Raises:
            Error: If the number is too large or too small to fit in Int.
        """
        # Int is 64-bit, so we need at most 2 words to represent it.
        # Int.MAX = 9_223_372_036_854_775_807 = 0x7FFF_FFFF_FFFF_FFFF
        if len(self.words) > 2:
            raise Error("BigInt2.to_int(): The number exceeds the size of Int")

        var magnitude: UInt64 = UInt64(self.words[0])
        if len(self.words) == 2:
            magnitude += UInt64(self.words[1]) << 32

        if self.sign:
            # Negative: check against Int.MIN magnitude (2^63)
            if magnitude > UInt64(9_223_372_036_854_775_808):
                raise Error(
                    "BigInt2.to_int(): The number exceeds the size of Int"
                )
            if magnitude == UInt64(9_223_372_036_854_775_808):
                return Int.MIN
            return -Int(magnitude)
        else:
            # Positive: check against Int.MAX (2^63 - 1)
            if magnitude > UInt64(9_223_372_036_854_775_807):
                raise Error(
                    "BigInt2.to_int(): The number exceeds the size of Int"
                )
            return Int(magnitude)

    fn to_bigint10(self) -> BigInt10:
        """Converts the BigInt2 to a base-10^9 BigInt10.

        Returns:
            The BigInt10 (base-10^9) representation with the same value.
        """
        if self.is_zero():
            return BigInt10()

        # Convert from base 2^32 to base 10^9 using repeated division
        var dividend = self.copy()
        var decimal_words = List[UInt32]()

        while not dividend.is_zero():
            var remainder: UInt64 = 0
            for i in range(len(dividend.words) - 1, -1, -1):
                var temp = (remainder << 32) + UInt64(dividend.words[i])
                dividend.words[i] = UInt32(temp // BigUInt.BASE)
                remainder = temp % BigUInt.BASE

            # Remove leading zeros from dividend
            while len(dividend.words) > 1 and dividend.words[-1] == 0:
                dividend.words.shrink(len(dividend.words) - 1)

            decimal_words.append(UInt32(remainder))

        return BigInt10(raw_words=decimal_words^, sign=self.sign)

    fn to_decimal_string(self, line_width: Int = 0) -> String:
        """Returns the decimal string representation of the BigInt2.

        Uses divide-and-conquer base conversion for large numbers (O(M(n)·log n))
        and simple repeated division by 10^9 for small numbers (O(n²)).

        Args:
            line_width: The maximum line width for the string representation.
                Default is 0, which means no line width limit.

        Returns:
            The decimal string (e.g. "-12345").
        """
        if self.is_zero():
            return String("0")

        # Get effective word count (excluding leading zeros)
        var eff_words = len(self.words)
        while eff_words > 1 and self.words[eff_words - 1] == 0:
            eff_words -= 1

        # Choose conversion strategy based on magnitude size
        var magnitude_str: String
        if eff_words <= _DC_TO_STR_ENTRY_THRESHOLD:
            magnitude_str = _magnitude_to_decimal_simple(self.words, eff_words)
        else:
            try:
                magnitude_str = _magnitude_to_decimal_dc(self.words, eff_words)
            except e:
                # Fallback to simple O(n²) method if D&C raises an Error
                magnitude_str = _magnitude_to_decimal_simple(
                    self.words, eff_words
                )

        var result: String
        if self.sign:
            result = String("-") + magnitude_str
        else:
            result = magnitude_str^

        if line_width > 0:
            var start = 0
            var end = line_width
            var lines = List[String](capacity=len(result) // line_width + 1)
            while end < len(result):
                lines.append(String(result[start:end]))
                start = end
                end += line_width
            lines.append(String(result[start:]))
            result = String("\n").join(lines^)

        return result^

    fn to_string_with_separators(self, separator: String = "_") -> String:
        """Returns string representation of the BigInt2 with separators.

        Args:
            separator: The separator string. Default is "_".

        Returns:
            The string representation of the BigInt2 with separators.
        """

        var result = self.to_decimal_string()
        var start_idx = 0
        if self.sign:
            start_idx = 1  # Skip the minus sign

        var digits_part = String(result[start_idx:])
        var end = len(digits_part)
        var start = end - 3
        var blocks = List[String](capacity=len(digits_part) // 3 + 1)
        while start > 0:
            blocks.append(String(digits_part[start:end]))
            end = start
            start = end - 3
        blocks.append(String(digits_part[0:end]))
        blocks.reverse()
        var formatted = separator.join(blocks)

        if self.sign:
            return String("-") + formatted
        return formatted^

    fn to_hex_string(self) -> String:
        """Returns a hexadecimal string representation of the BigInt2.

        Returns:
            The hexadecimal string (e.g. "0x1A2B3C").
        """
        if self.is_zero():
            return "0x0"

        var result = String()
        if self.sign:
            result += "-"
        result += "0x"

        var first_word = True
        for i in range(len(self.words) - 1, -1, -1):
            var word = self.words[i]
            if first_word:
                if word != 0:
                    result += hex(word)[2:]
                    first_word = False
            else:
                var h = hex(word)[2:]
                for _ in range(8 - len(h)):
                    result += "0"
                result += h

        if first_word:
            result += "0"

        return result

    fn to_binary_string(self) -> String:
        """Returns a binary string representation of the BigInt2.

        Returns:
            The binary string (e.g. "0b110101").
        """
        if self.is_zero():
            return "0b0"

        var result = String()
        if self.sign:
            result += "-"
        result += "0b"

        var first_word = True
        for i in range(len(self.words) - 1, -1, -1):
            var word = self.words[i]
            if first_word:
                if word != 0:
                    result += bin(word)[2:]
                    first_word = False
            else:
                var b = bin(word)[2:]
                for _ in range(32 - len(b)):
                    result += "0"
                result += b

        if first_word:
            result += "0"

        return result

    # ===------------------------------------------------------------------=== #
    # Unary arithmetic dunders
    # ===------------------------------------------------------------------=== #

    fn __neg__(self) -> Self:
        """Returns the negation of the BigInt2."""
        if self.is_zero():
            return Self()
        return Self(raw_words=self.words.copy(), sign=not self.sign)

    fn __abs__(self) -> Self:
        """Returns the absolute value of the BigInt2."""
        return Self(raw_words=self.words.copy(), sign=False)

    # ===------------------------------------------------------------------=== #
    # Basic binary arithmetic operation dunders
    # These methods are called to implement the binary arithmetic operations
    # (+, -, *, @, /, //, %, divmod(), pow(), **, <<, >>, &, ^, |)
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __add__(self, other: Self) -> Self:
        return decimojo.bigint2.arithmetics.add(self, other)

    @always_inline
    fn __sub__(self, other: Self) -> Self:
        return decimojo.bigint2.arithmetics.subtract(self, other)

    @always_inline
    fn __mul__(self, other: Self) -> Self:
        return decimojo.bigint2.arithmetics.multiply(self, other)

    @always_inline
    fn __floordiv__(self, other: Self) raises -> Self:
        try:
            return decimojo.bigint2.arithmetics.floor_divide(self, other)
        except e:
            raise Error(
                DeciMojoError(
                    message=None,
                    function="BigInt2.__floordiv__()",
                    file="src/decimojo/bigint2/bigint2.mojo",
                    previous_error=e^,
                )
            )

    @always_inline
    fn __mod__(self, other: Self) raises -> Self:
        try:
            return decimojo.bigint2.arithmetics.floor_modulo(self, other)
        except e:
            raise Error(
                DeciMojoError(
                    message=None,
                    function="BigInt2.__mod__()",
                    file="src/decimojo/bigint2/bigint2.mojo",
                    previous_error=e^,
                )
            )

    @always_inline
    fn __divmod__(self, other: Self) raises -> Tuple[Self, Self]:
        try:
            return decimojo.bigint2.arithmetics.floor_divmod(self, other)
        except e:
            raise Error(
                DeciMojoError(
                    message=None,
                    function="BigInt2.__divmod__()",
                    file="src/decimojo/bigint2/bigint2.mojo",
                    previous_error=e^,
                )
            )

    @always_inline
    fn __pow__(self, exponent: Self) raises -> Self:
        return self.power(exponent)

    @always_inline
    fn __pow__(self, exponent: Int) raises -> Self:
        return self.power(exponent)

    @always_inline
    fn __lshift__(self, shift: Int) -> Self:
        """Returns self << shift (multiply by 2^shift)."""
        return decimojo.bigint2.arithmetics.left_shift(self, shift)

    @always_inline
    fn __rshift__(self, shift: Int) -> Self:
        """Returns self >> shift (floor divide by 2^shift)."""
        return decimojo.bigint2.arithmetics.right_shift(self, shift)

    # ===------------------------------------------------------------------=== #
    # Basic binary right-side arithmetic operation dunders
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __radd__(self, other: Self) -> Self:
        return decimojo.bigint2.arithmetics.add(self, other)

    @always_inline
    fn __rsub__(self, other: Self) -> Self:
        return decimojo.bigint2.arithmetics.subtract(other, self)

    @always_inline
    fn __rmul__(self, other: Self) -> Self:
        return decimojo.bigint2.arithmetics.multiply(self, other)

    @always_inline
    fn __rfloordiv__(self, other: Self) raises -> Self:
        return decimojo.bigint2.arithmetics.floor_divide(other, self)

    @always_inline
    fn __rmod__(self, other: Self) raises -> Self:
        return decimojo.bigint2.arithmetics.floor_modulo(other, self)

    @always_inline
    fn __rdivmod__(self, other: Self) raises -> Tuple[Self, Self]:
        return decimojo.bigint2.arithmetics.floor_divmod(other, self)

    @always_inline
    fn __rpow__(self, base: Self) raises -> Self:
        return base.power(self)

    # ===------------------------------------------------------------------=== #
    # Basic binary augmented arithmetic assignments dunders
    # (+=, -=, *=, //=, %=)
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __iadd__(mut self, other: Self):
        """True in-place addition: mutates self.words directly."""
        decimojo.bigint2.arithmetics.add_inplace(self, other)

    @always_inline
    fn __iadd__(mut self, other: Int):
        """True in-place addition with Int: mutates self.words directly."""
        decimojo.bigint2.arithmetics.add_inplace_int(self, other)

    @always_inline
    fn __isub__(mut self, other: Self):
        """True in-place subtraction: mutates self.words directly."""
        decimojo.bigint2.arithmetics.subtract_inplace(self, other)

    @always_inline
    fn __imul__(mut self, other: Self):
        """True in-place multiplication: computes product into self.words."""
        decimojo.bigint2.arithmetics.multiply_inplace(self, other)

    @always_inline
    fn __ifloordiv__(mut self, other: Self) raises:
        """True in-place floor division: moves quotient into self.words."""
        decimojo.bigint2.arithmetics.floor_divide_inplace(self, other)

    @always_inline
    fn __imod__(mut self, other: Self) raises:
        """True in-place modulo: moves remainder into self.words."""
        decimojo.bigint2.arithmetics.floor_modulo_inplace(self, other)

    @always_inline
    fn __ilshift__(mut self, shift: Int):
        """True in-place left shift: mutates self.words directly."""
        decimojo.bigint2.arithmetics.left_shift_inplace(self, shift)

    @always_inline
    fn __irshift__(mut self, shift: Int):
        """True in-place right shift: mutates self.words directly."""
        decimojo.bigint2.arithmetics.right_shift_inplace(self, shift)

    # ===------------------------------------------------------------------=== #
    # Basic binary comparison operation dunders
    # __gt__, __ge__, __lt__, __le__, __eq__, __ne__
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __gt__(self, other: Self) -> Bool:
        """Returns True if self > other."""
        return decimojo.bigint2.comparison.greater(self, other)

    @always_inline
    fn __gt__(self, other: Int) -> Bool:
        """Returns True if self > other."""
        return decimojo.bigint2.comparison.greater(self, Self.from_int(other))

    @always_inline
    fn __ge__(self, other: Self) -> Bool:
        """Returns True if self >= other."""
        return decimojo.bigint2.comparison.greater_equal(self, other)

    @always_inline
    fn __ge__(self, other: Int) -> Bool:
        """Returns True if self >= other."""
        return decimojo.bigint2.comparison.greater_equal(
            self, Self.from_int(other)
        )

    @always_inline
    fn __lt__(self, other: Self) -> Bool:
        """Returns True if self < other."""
        return decimojo.bigint2.comparison.less(self, other)

    @always_inline
    fn __lt__(self, other: Int) -> Bool:
        """Returns True if self < other."""
        return decimojo.bigint2.comparison.less(self, Self.from_int(other))

    @always_inline
    fn __le__(self, other: Self) -> Bool:
        """Returns True if self <= other."""
        return decimojo.bigint2.comparison.less_equal(self, other)

    @always_inline
    fn __le__(self, other: Int) -> Bool:
        """Returns True if self <= other."""
        return decimojo.bigint2.comparison.less_equal(
            self, Self.from_int(other)
        )

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Returns True if self == other."""
        return decimojo.bigint2.comparison.equal(self, other)

    @always_inline
    fn __eq__(self, other: Int) -> Bool:
        """Returns True if self == other."""
        return decimojo.bigint2.comparison.equal(self, Self.from_int(other))

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Returns True if self != other."""
        return decimojo.bigint2.comparison.not_equal(self, other)

    @always_inline
    fn __ne__(self, other: Int) -> Bool:
        """Returns True if self != other."""
        return decimojo.bigint2.comparison.not_equal(self, Self.from_int(other))

    # ===------------------------------------------------------------------=== #
    # Mathematical methods that do not implement a trait (not a dunder)
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn truncate_divide(self, other: Self) raises -> Self:
        """Performs a truncated division of two BigInt2 numbers.
        See `truncate_divide()` for more information.
        """
        return decimojo.bigint2.arithmetics.truncate_divide(self, other)

    @always_inline
    fn floor_modulo(self, other: Self) raises -> Self:
        """Performs a floor modulo of two BigInt2 numbers.
        See `floor_modulo()` for more information.
        """
        return decimojo.bigint2.arithmetics.floor_modulo(self, other)

    @always_inline
    fn truncate_modulo(self, other: Self) raises -> Self:
        """Performs a truncated modulo of two BigInt2 numbers.
        See `truncate_modulo()` for more information.
        """
        return decimojo.bigint2.arithmetics.truncate_modulo(self, other)

    fn power(self, exponent: Int) raises -> Self:
        """Raises the BigInt2 to the power of an integer exponent.

        Args:
            exponent: The non-negative exponent.

        Returns:
            The result of self raised to the given exponent.

        Raises:
            Error: If the exponent is negative.
        """
        return decimojo.bigint2.arithmetics.power(self, exponent)

    fn power(self, exponent: Self) raises -> Self:
        """Raises the BigInt2 to the power of another BigInt2.

        Args:
            exponent: The exponent (must be non-negative and fit in Int).

        Returns:
            The result of self raised to the given exponent.

        Raises:
            Error: If the exponent is negative or too large.
        """
        if exponent.is_negative():
            raise Error("BigInt2.power(): Exponent must be non-negative")
        var exp_int: Int
        try:
            exp_int = exponent.to_int()
        except e:
            raise Error("BigInt2.power(): Exponent too large to fit in Int")
        return self.power(exp_int)

    fn sqrt(self) raises -> Self:
        """Returns the integer square root of this BigInt2.

        The result is the largest integer y such that y * y <= self
        (for non-negative self). Only defined for non-negative values.

        Returns:
            The integer square root.

        Raises:
            Error: If the value is negative.
        """
        return decimojo.bigint2.exponential.sqrt(self)

    fn isqrt(self) raises -> Self:
        """Returns the integer square root of this BigInt2.
        It is equal to `sqrt()`.

        Returns:
            The integer square root.

        Raises:
            Error: If the value is negative.
        """
        return decimojo.bigint2.exponential.sqrt(self)

    @always_inline
    fn compare_magnitudes(self, other: Self) -> Int8:
        """Compares the magnitudes (absolute values) of two BigInt2 numbers.
        See `compare_magnitudes()` for more information.
        """
        return decimojo.bigint2.comparison.compare_magnitudes(self, other)

    @always_inline
    fn compare(self, other: Self) -> Int8:
        """Compares two BigInt2 numbers.
        See `compare()` for more information.
        """
        return decimojo.bigint2.comparison.compare(self, other)

    # ===------------------------------------------------------------------=== #
    # Bitwise operations
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __and__(self, other: Self) -> Self:
        """Returns self & other (bitwise AND, Python two's complement semantics).
        """
        return decimojo.bigint2.bitwise.bitwise_and(self, other)

    @always_inline
    fn __and__(self, other: Int) -> Self:
        """Returns self & other where other is an Int."""
        return decimojo.bigint2.bitwise.bitwise_and(self, Self(other))

    @always_inline
    fn __or__(self, other: Self) -> Self:
        """Returns self | other (bitwise OR, Python two's complement semantics).
        """
        return decimojo.bigint2.bitwise.bitwise_or(self, other)

    @always_inline
    fn __or__(self, other: Int) -> Self:
        """Returns self | other where other is an Int."""
        return decimojo.bigint2.bitwise.bitwise_or(self, Self(other))

    @always_inline
    fn __xor__(self, other: Self) -> Self:
        """Returns self ^ other (bitwise XOR, Python two's complement semantics).
        """
        return decimojo.bigint2.bitwise.bitwise_xor(self, other)

    @always_inline
    fn __xor__(self, other: Int) -> Self:
        """Returns self ^ other where other is an Int."""
        return decimojo.bigint2.bitwise.bitwise_xor(self, Self(other))

    @always_inline
    fn __invert__(self) -> Self:
        """Returns ~self (bitwise NOT, Python two's complement semantics)."""
        return decimojo.bigint2.bitwise.bitwise_not(self)

    @always_inline
    fn __iand__(mut self, other: Self):
        """True in-place bitwise AND: mutates self.words directly."""
        decimojo.bigint2.bitwise.bitwise_and_inplace(self, other)

    @always_inline
    fn __ior__(mut self, other: Self):
        """True in-place bitwise OR: mutates self.words directly."""
        decimojo.bigint2.bitwise.bitwise_or_inplace(self, other)

    @always_inline
    fn __ixor__(mut self, other: Self):
        """True in-place bitwise XOR: mutates self.words directly."""
        decimojo.bigint2.bitwise.bitwise_xor_inplace(self, other)

    # ===------------------------------------------------------------------=== #
    # Instance query methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn is_zero(self) -> Bool:
        """Returns True if the value is zero."""
        if len(self.words) == 1 and self.words[0] == 0:
            return True
        for word in self.words:
            if word != 0:
                return False
        return True

    @always_inline
    fn is_negative(self) -> Bool:
        """Returns True if the value is strictly negative."""
        return self.sign and not self.is_zero()

    @always_inline
    fn is_positive(self) -> Bool:
        """Returns True if the value is strictly positive."""
        return not self.sign and not self.is_zero()

    fn is_one(self) -> Bool:
        """Returns True if the value is exactly 1."""
        return not self.sign and len(self.words) == 1 and self.words[0] == 1

    fn is_one_or_minus_one(self) -> Bool:
        """Returns True if the value is 1 or -1."""
        return len(self.words) == 1 and self.words[0] == 1

    fn bit_length(self) -> Int:
        """Returns the number of bits needed to represent the magnitude,
        excluding leading zeros.

        Returns:
            The position of the highest set bit, or 0 if the value is zero.
        """
        if self.is_zero():
            return 0

        var n_words = len(self.words)
        var msw = self.words[n_words - 1]

        # Count bits in the most significant word
        var bits_in_msw = 32
        var probe: UInt32 = 1 << 31
        while probe != 0 and (msw & probe) == 0:
            bits_in_msw -= 1
            probe >>= 1

        return (n_words - 1) * 32 + bits_in_msw

    fn number_of_words(self) -> Int:
        """Returns the number of words in the magnitude."""
        return len(self.words)

    fn number_of_digits(self) -> Int:
        """Returns the number of decimal digits in the magnitude.

        Notes:
            Zero has 1 digit.
        """
        if self.is_zero():
            return 1

        # Convert to BigInt10 and use its digit counting
        return self.to_bigint10().magnitude.number_of_digits()

    # ===------------------------------------------------------------------=== #
    # Internal utility methods
    # ===------------------------------------------------------------------=== #

    fn copy(self) -> Self:
        """Returns a deep copy of this BigInt2."""
        var new_words = List[UInt32](capacity=len(self.words))
        for word in self.words:
            new_words.append(word)
        return Self(raw_words=new_words^, sign=self.sign)

    fn _normalize(mut self):
        """Strips leading zero words and normalizes -0 to +0."""
        while len(self.words) > 1 and self.words[-1] == 0:
            self.words.shrink(len(self.words) - 1)

        # Normalize -0 to +0
        if self.is_zero():
            self.sign = False

    fn print_internal_representation(self):
        """Prints the internal representation details."""
        print("\nInternal Representation Details of BigInt2")
        print("------------------------------------------------")
        print("decimal:        " + self.to_decimal_string())
        print("hex:            " + self.to_hex_string())
        print(
            "sign:           "
            + String("negative" if self.sign else "non-negative")
        )
        print("words:          " + String(len(self.words)))
        for i in range(len(self.words)):
            var ndigits = 1
            if i >= 100:
                ndigits = 3
            elif i >= 10:
                ndigits = 2
            print(
                "  word ",
                i,
                ":",
                " " * (6 - ndigits),
                "0x",
                hex(self.words[i])[2:].rjust(8, fillchar="0"),
                "  (",
                self.words[i],
                ")",
                sep="",
            )
        print("------------------------------------------------")


# ===----------------------------------------------------------------------=== #
# Module-level private helpers for from_string
# These operate on the magnitude words only (sign is handled by caller).
# ===----------------------------------------------------------------------=== #


fn _multiply_inplace_by_uint32(mut x: BigInt2, y: UInt32):
    """Multiplies a BigInt2 magnitude by a UInt32 scalar in-place.

    This is used internally by from_string() during base conversion.

    Args:
        x: The BigInt2 to multiply (modified in-place).
        y: The UInt32 scalar multiplier.
    """
    if y == 0:
        x.words = [UInt32(0)]
        x.sign = False
        return
    if y == 1:
        return

    var carry: UInt64 = 0
    for i in range(len(x.words)):
        var product = UInt64(x.words[i]) * UInt64(y) + carry
        x.words[i] = UInt32(product & 0xFFFF_FFFF)
        carry = product >> 32

    if carry > 0:
        x.words.append(UInt32(carry))


fn _add_inplace_by_uint32(mut x: BigInt2, y: UInt32):
    """Adds a UInt32 value to a BigInt2 magnitude in-place.

    This is used internally by from_string() during base conversion.

    Args:
        x: The BigInt2 to add to (modified in-place).
        y: The UInt32 value to add.
    """
    if y == 0:
        return

    var carry: UInt64 = UInt64(y)
    for i in range(len(x.words)):
        if carry == 0:
            break
        var sum = UInt64(x.words[i]) + carry
        x.words[i] = UInt32(sum & 0xFFFF_FFFF)
        carry = sum >> 32

    if carry > 0:
        x.words.append(UInt32(carry))


fn _multiply_add_inplace(mut x: BigInt2, mul: UInt32, add: UInt32):
    """Computes x = x * mul + add in a single pass over the word array.

    Fuses the multiply-by-scalar and add-scalar operations into one O(n) pass
    instead of two separate O(n) passes, halving memory traffic. This is the
    inner loop of the simple base-conversion path (9 digits at a time).

    Correctness: at each word position i,
        product = x.words[i] * mul + carry
    where carry starts at `add` and propagates upward. This correctly computes
    x * mul + add because the carry chain handles both the multiplication
    carry and the initial addend.

    Overflow safety: product <= (2^32-1)*(2^32-1) + carry. Since carry < 2^32
    after the first step, product < 2^64, fitting in UInt64.

    Args:
        x: The BigInt2 to modify in-place.
        mul: The UInt32 multiplier (e.g. 10^9).
        add: The UInt32 addend (e.g. a 9-digit chunk value).
    """
    if mul == 0:
        x.words = [UInt32(add)]
        x.sign = False
        return

    var carry: UInt64 = UInt64(add)
    for i in range(len(x.words)):
        var product = UInt64(x.words[i]) * UInt64(mul) + carry
        x.words[i] = UInt32(product & 0xFFFF_FFFF)
        carry = product >> 32

    if carry > 0:
        x.words.append(UInt32(carry))


# ===----------------------------------------------------------------------=== #
# Divide-and-conquer base conversion (decimal string → binary)
# ===----------------------------------------------------------------------=== #

# Thresholds for D&C from_string, measured in decimal digit count.
# The simple multiply-and-add method has very low constant factors
# (sequential UInt32 operations), so D&C only wins at much larger sizes
# than for to_string (where the saved divisions are each expensive).
# Entry threshold: only enter D&C when the digit count is large enough
# that the O(n²) simple method is significantly slower than the O(M(n)·log n)
# D&C method despite the power-table construction overhead.
# Base threshold: within the recursion, switch to simple method.
comptime _DC_FROM_STR_ENTRY_THRESHOLD = 10000
comptime _DC_FROM_STR_BASE_THRESHOLD = 256


fn _from_decimal_digits_simple(
    digits: List[UInt8], start: Int, end: Int
) -> BigInt2:
    """Converts a range of digit values to a BigInt2 using the simple
    O(n²) multiply-and-add method (9 digits at a time).

    Optimizations over the naive approach:
    - Pre-allocates the word array to its maximum possible size, avoiding
      all dynamic growth (append / reallocation) during conversion.
    - Handles the first (possibly shorter) chunk separately so the main
      loop always processes exactly 9 digits with a compile-time constant
      10^9 multiplier — no inner loop to compute 10^chunk_size.
    - Uses raw pointer access for both the digit array and the word array
      to eliminate bounds-checking overhead in the hot inner loop.
    - Tracks the live word count in a local variable, trimming once at end.

    Args:
        digits: List of digit values (0-9).
        start: Start index (inclusive) in the digits list.
        end: End index (exclusive) in the digits list.

    Returns:
        The unsigned BigInt2 value (sign is False).
    """
    if start >= end:
        return BigInt2()

    var digit_count = end - start

    # ---- Fast path: ≤ 9 digits → single UInt32 word, no allocation ----
    if digit_count <= 9:
        var dp = digits._data + start
        var val: UInt32 = UInt32(dp[0])
        for j in range(1, digit_count):
            val = val * 10 + UInt32(dp[j])
        var result = BigInt2()
        result.words[0] = val
        return result^

    # ---- Fast path: 10–19 digits → parse into UInt64, at most 2 words ----
    if digit_count <= 19:
        var dp = digits._data + start
        var val: UInt64 = UInt64(dp[0])
        for j in range(1, digit_count):
            val = val * 10 + UInt64(dp[j])
        var result = BigInt2()
        result.words[0] = UInt32(val & 0xFFFF_FFFF)
        var high_word = UInt32(val >> 32)
        if high_word > 0:
            result.words.append(high_word)
        return result^

    # ---- General path: pre-allocate and multiply-add by 10^9 chunks ----

    # Pre-allocate words: ceil(digit_count * log2(10) / 32) + 2.
    # 107/1024 ≈ 0.10449 > log2(10)/32 ≈ 0.10381, so always sufficient.
    # The +2 guarantees room for a carry word at the end of every iteration.
    var max_words = (digit_count * 107 + 1023) // 1024 + 2

    var result = BigInt2()
    result.words = List[UInt32](capacity=max_words)
    result.words.resize(unsafe_uninit_length=max_words)
    var wp = result.words._data  # stable pointer: no reallocation occurs

    # Handle first chunk (1–9 digits) to align the rest to 9-digit boundaries.
    var first_chunk = digit_count % 9
    if first_chunk == 0:
        first_chunk = 9

    var dp = digits._data + start
    var chunk_val: UInt32 = UInt32(dp[0])
    for j in range(1, first_chunk):
        chunk_val = chunk_val * 10 + UInt32(dp[j])
    dp += first_chunk

    wp[0] = chunk_val
    var word_count: Int = 1
    var remaining = digit_count - first_chunk

    # Main loop: full 9-digit chunks with constant multiplier 10^9.
    comptime MUL9: UInt64 = 1_000_000_000

    while remaining > 0:
        # Parse 9 digit values → UInt32 chunk
        var cv: UInt32 = UInt32(dp[0])
        for j in range(1, 9):
            cv = cv * 10 + UInt32(dp[j])
        dp += 9
        remaining -= 9

        # Fused multiply-add: result = result * 10^9 + cv  (single O(n) pass)
        var carry: UInt64 = UInt64(cv)
        for k in range(word_count):
            var product = UInt64(wp[k]) * MUL9 + carry
            wp[k] = UInt32(product & 0xFFFF_FFFF)
            carry = product >> 32
        if carry > 0:
            wp[word_count] = UInt32(carry)
            word_count += 1

    # Trim pre-allocated words to the actual live word count.
    while len(result.words) > word_count:
        result.words.shrink(len(result.words) - 1)

    return result^


fn _from_decimal_digits_dc(
    digits: List[UInt8], start: Int, end: Int
) raises -> BigInt2:
    """Converts a range of digit values to a BigInt2 using
    divide-and-conquer base conversion. Complexity: O(M(n) · log n)
    where M(n) is the multiplication cost.

    Algorithm:
    1. Precompute a power table: powers[k] = 10^(2^k) as BigInt2 values.
    2. **Balanced split**: choose the largest power-of-2 boundary ≤ digit_count/2.
       This keeps both halves close in size, which is optimal for Karatsuba
       multiplication (balanced operands give the best O(n^1.585) constant).
    3. Recursively convert both halves.
    4. Combine: result = high * powers[k] + low.

    The balanced split also reduces the power-table size: we only build up
    to floor(log2(digit_count/2)) instead of ceil(log2(digit_count)), saving
    one expensive squaring at the top level.

    Args:
        digits: List of digit values (0-9).
        start: Start index (inclusive) in the digits list.
        end: End index (exclusive) in the digits list.

    Returns:
        The unsigned BigInt2 value (sign is False).
    """
    var digit_count = end - start

    # For balanced D&C, the top-level split uses 2^k ≤ digit_count/2.
    # We only need power table entries up to that level, saving one
    # expensive squaring compared to the "largest 2^k < digit_count" approach.
    var half = digit_count >> 1
    var top_level = 0
    var tmp = half
    while tmp > 0:
        tmp >>= 1
        top_level += 1
    top_level -= 1  # floor(log2(half)): 2^top_level ≤ half < 2^(top_level+1)

    # Build power table: powers[k] = 10^(2^k). Indices 0..top_level.
    var num_powers = top_level + 1
    var power_table = List[BigInt2](capacity=num_powers)
    power_table.append(BigInt2(10))
    for k in range(1, num_powers):
        # Compute 10^(2^k) = (10^(2^(k-1)))^2 directly from the table.
        var sq = power_table[k - 1] * power_table[k - 1]
        power_table.append(sq^)

    # Run the recursive D&C conversion
    return _dc_from_str_recursive(digits, start, end, power_table, top_level)


fn _dc_from_str_recursive(
    digits: List[UInt8],
    start: Int,
    end: Int,
    power_table: List[BigInt2],
    max_level: Int,
) raises -> BigInt2:
    """Recursively converts a range of digit values to BigInt2
    using the precomputed power table with balanced splitting.

    At each level, splits the digit range into high and low parts where
    the low part has 2^level digits (the largest power-of-2 ≤ digit_count/2),
    then:
        result = high * 10^(2^level) + low

    The balanced split ensures high and low are within a 2:1 ratio, keeping
    the combine multiplication efficient under Karatsuba.

    Note on max_level: the high part receives `level` (not `level - 1`)
    because high ≥ digit_count/2, so it may legitimately need the same level.
    The low part receives `level - 1` since it has exactly 2^level digits.

    Args:
        digits: List of digit values (0-9).
        start: Start index (inclusive).
        end: End index (exclusive).
        power_table: Precomputed table where powers[k] = 10^(2^k).
        max_level: Maximum level accessible in power_table for this sub-problem.

    Returns:
        The unsigned BigInt2 value for digits[start:end].
    """
    var digit_count = end - start

    # Base case: small enough for simple O(n²) conversion
    if digit_count <= _DC_FROM_STR_BASE_THRESHOLD:
        return _from_decimal_digits_simple(digits, start, end)

    # Find the largest level k such that 2^k ≤ digit_count / 2.
    # This balanced split keeps operands close in size for Karatsuba.
    var level = min(max_level, len(power_table) - 1)
    var half = digit_count >> 1
    while level >= 0 and (1 << level) > half:
        level -= 1

    if level < 0:
        # digit_count ≤ 2 (can't split meaningfully), use simple method
        return _from_decimal_digits_simple(digits, start, end)

    # Split: low part has exactly 2^level digits, high part gets the rest.
    var low_len = 1 << level
    var split = end - low_len

    # Recursively convert both halves.
    # High part may need the same `level` (since high ≥ digit_count/2),
    # so pass `level` rather than `level - 1`.
    var high = _dc_from_str_recursive(digits, start, split, power_table, level)
    var low = _dc_from_str_recursive(digits, split, end, power_table, level - 1)

    # Combine: result = high * 10^(2^level) + low
    # Use _add_magnitudes_inplace directly to avoid BigInt2.__iadd__ overhead
    # (which creates a new BigInt2 via arithmetics.add).
    var result = high * power_table[level]
    decimojo.bigint2.arithmetics._add_magnitudes_inplace(
        result.words, low.words
    )
    return result^


# ===----------------------------------------------------------------------=== #
# Divide-and-conquer base conversion (binary → decimal string)
# ===----------------------------------------------------------------------=== #

# The threshold (in UInt32 words) below which we use the simple O(n²) method
# of repeated division by 10^9. Above this, the D&C method is used.
# D&C only wins when the internal divisions can use the sub-quadratic
# Burnikel-Ziegler algorithm (CUTOFF_BURNIKEL_ZIEGLER = 64 words).
# Since the D&C divisor is roughly half the dividend, we need the dividend
# to be ≥ 2 × 64 = 128 words for B-Z to kick in at the first split.
#
# We use TWO thresholds:
# - _DC_TO_STR_ENTRY_THRESHOLD (128): gates the top-level decision to enter D&C
#   (~1230 decimal digits; below this the simple O(n²) path is faster)
# - _DC_TO_STR_BASE_THRESHOLD (64): base-case size within the recursion
#   (~616 decimal digits; recursion bottoms out to simple path here)
comptime _DC_TO_STR_ENTRY_THRESHOLD = 128
comptime _DC_TO_STR_BASE_THRESHOLD = 64

# Base for extracting 9-digit decimal chunks in the simple conversion path.
# Same numerical value as BigUInt.BASE, but defined locally to avoid
# coupling the binary→decimal conversion logic to the BigUInt type.
comptime _DECIMAL_CHUNK_BASE: UInt64 = 1_000_000_000


fn _magnitude_to_decimal_simple(words: List[UInt32], eff_words: Int) -> String:
    """Converts a magnitude (unsigned word list) to a decimal string using
    the simple O(n²) method of repeated division by 10^9.

    Optimizations over naive approach:
    - Divides by 10^9, collecting base-10^9 chunks, then writes digits
      to a byte buffer in one pass (no string concatenation).
    - Tracks effective dividend length (`div_len`) instead of scanning
      for is_zero.
    - Uses `unsafe_ptr()` for the inner division loop.

    Args:
        words: The magnitude in little-endian UInt32 words.
        eff_words: Effective number of words (excluding trailing zeros).

    Returns:
        The unsigned decimal string (no sign prefix).
    """
    if eff_words == 1 and words[0] == 0:
        return String("0")

    # Fast path for single-word values.
    if eff_words == 1:
        return String(Int(words[0]))

    # Fast path for two-word values (fits in UInt64).
    if eff_words == 2:
        var val = (UInt64(words[1]) << 32) | UInt64(words[0])
        return String(val)

    # Allocate dividend buffer and get raw pointer for fast inner loop.
    var dividend = List[UInt32](capacity=eff_words)
    for i in range(eff_words):
        dividend.append(words[i])
    var dp = dividend.unsafe_ptr()

    # Estimate number of 9-digit chunks: ceil(bits * log10(2) / 9) + 1.
    var est_chunks = (eff_words * 32 * 9 + 268) // 269 + 1

    # Extract base-10^9 chunks via repeated division.
    var chunks = List[UInt32](capacity=est_chunks)
    var div_len = eff_words

    while div_len > 0:
        var remainder: UInt64 = 0
        for i in range(div_len - 1, -1, -1):
            var temp = (remainder << 32) + UInt64(dp[i])
            dp[i] = UInt32(temp // _DECIMAL_CHUNK_BASE)
            remainder = temp % _DECIMAL_CHUNK_BASE

        while div_len > 0 and dp[div_len - 1] == 0:
            div_len -= 1

        chunks.append(UInt32(remainder))

    var num_chunks = len(chunks)
    if num_chunks == 0:
        return String("0")

    # --- Build the decimal string in a byte buffer ---
    var max_digits = num_chunks * 9
    var buf = List[UInt8](capacity=max_digits + 1)

    # Most-significant chunk: no zero-padding.
    var msb = Int(chunks[num_chunks - 1])
    var msb_digits = InlineArray[UInt8, 10](fill=0)
    var msb_len = 0
    if msb == 0:
        buf.append(48)  # '0'
    else:
        var v = msb
        while v > 0:
            msb_digits[msb_len] = UInt8(v % 10) + 48
            msb_len += 1
            v //= 10
        for j in range(msb_len - 1, -1, -1):
            buf.append(msb_digits[j])

    # Remaining chunks: zero-padded to exactly 9 digits.
    for ci in range(num_chunks - 2, -1, -1):
        var val = Int(chunks[ci])
        var digits9 = InlineArray[UInt8, 9](fill=48)  # pre-fill '0'
        for d in range(9):
            digits9[8 - d] = UInt8(val % 10) + 48
            val //= 10
        for d in range(9):
            buf.append(digits9[d])

    return String(unsafe_from_utf8=buf^)


fn _magnitude_to_decimal_dc(
    words: List[UInt32], eff_words: Int
) raises -> String:
    """Converts a magnitude to a decimal string using divide-and-conquer
    base conversion. Complexity: O(M(n) · log n) where M(n) is the
    multiplication cost.

    Algorithm:
    1. Precompute a power table: powers[k] = 10^(2^k) as BigInt2 values.
    2. Find the largest k such that powers[k] ≤ the number.
    3. divmod(number, powers[k]) → (high, low).
    4. The low part has exactly 2^k decimal digits (zero-padded).
    5. Recursively convert high and low halves.

    Args:
        words: The magnitude in little-endian UInt32 words.
        eff_words: Effective number of words (excluding trailing zeros).

    Returns:
        The unsigned decimal string (no sign prefix).
    """
    # Estimate decimal digits from bit length
    # bit_length = (eff_words - 1) * 32 + bits_in_top_word
    var top_word = words[eff_words - 1]
    var bits_in_top = 32
    var probe: UInt32 = 1 << 31
    while probe != 0 and (top_word & probe) == 0:
        bits_in_top -= 1
        probe >>= 1
    var total_bits = (eff_words - 1) * 32 + bits_in_top

    # Conservative overestimate: digits <= floor(bits * log10(2)) + 1
    # log10(2) ≈ 0.30103 ≈ 78/259 (slightly over)
    var est_digits = (total_bits * 78 + 258) // 259 + 1

    # Find max_level such that 2^max_level >= est_digits.
    # We only need powers[0..max_level-1] because 10^(2^max_level) > n,
    # so the first split is always at a level < max_level.
    # Use bit-counting instead of `1 << max_level` to avoid overflow
    # when est_digits approaches the Int word size.
    var max_level = 0
    var tmp = est_digits - 1
    while tmp > 0:
        tmp >>= 1
        max_level += 1

    # Build power table: powers[k] = 10^(2^k) as BigInt2
    # powers[0] = 10^1, powers[1] = 10^2, powers[2] = 10^4, ...
    # Only build up to max_level - 1 (the highest level that can be used).
    var num_powers = max_level  # indices 0 to max_level - 1
    var power_table = List[BigInt2](capacity=num_powers)
    power_table.append(BigInt2(10))
    for k in range(1, num_powers):
        var sq = power_table[k - 1] * power_table[k - 1]
        power_table.append(sq^)

    # Create unsigned BigInt2 from the magnitude words
    var trimmed = List[UInt32](capacity=eff_words)
    for i in range(eff_words):
        trimmed.append(words[i])
    var n = BigInt2(raw_words=trimmed^, sign=False)

    # Run the recursive D&C conversion
    return _dc_to_str_recursive(n, power_table, num_powers - 1)


fn _dc_to_str_recursive(
    n: BigInt2,
    power_table: List[BigInt2],
    max_level: Int,
) raises -> String:
    """Recursively converts a non-negative BigInt2 to decimal string
    using the precomputed power table.

    At each level, divides by powers[level] = 10^(2^level) to split the number
    into a high part and a low part with exactly 2^level decimal digits.

    Args:
        n: The non-negative number to convert.
        power_table: Precomputed table where powers[k] = 10^(2^k).
        max_level: Maximum level accessible in power_table for this subproblem.

    Returns:
        The decimal string representation.
    """
    # Base case: small enough for simple O(n²) conversion
    var eff = len(n.words)
    while eff > 1 and n.words[eff - 1] == 0:
        eff -= 1

    if eff <= _DC_TO_STR_BASE_THRESHOLD:
        return _magnitude_to_decimal_simple(n.words, eff)

    # Find the largest level k such that powers[k] <= n
    var level = -1
    for k in range(min(max_level + 1, len(power_table))):
        if n >= power_table[k]:
            level = k
        else:
            break

    if level < 0:
        # n < 10, use simple method
        return _magnitude_to_decimal_simple(n.words, eff)

    # Divide n by powers[level] = 10^(2^level)
    var qr = decimojo.bigint2.arithmetics.floor_divmod(n, power_table[level])
    var q = qr[0].copy()
    var r = qr[1].copy()

    # The low part has exactly 2^level decimal digits
    var low_width = 1 << level

    # Recurse on high and low parts
    var high_str = _dc_to_str_recursive(q, power_table, level - 1)
    var low_str = _dc_to_str_recursive(r, power_table, level - 1)

    # Zero-pad low_str to exactly low_width digits.
    # Build the result in a pre-allocated byte buffer to avoid O(n) concatenations.
    var padding = low_width - len(low_str)
    if padding <= 0:
        return high_str + low_str

    var total_len = len(high_str) + padding + len(low_str)
    var buf = List[UInt8](capacity=total_len)
    # Copy high_str bytes
    for i in range(len(high_str)):
        buf.append(high_str.unsafe_ptr()[i])
    # Write zero padding
    for _ in range(padding):
        buf.append(48)  # ASCII '0'
    # Copy low_str bytes
    for i in range(len(low_str)):
        buf.append(low_str.unsafe_ptr()[i])

    return String(unsafe_from_utf8=buf^)
