"""
DataTypes
=========

This module contains classes for fixed-size data types. These classes are used to enforce the size of the data type and to provide a way to validate the value of the data type.
This also ensures the size (in bites) of the data type is fixed and know.
This can also optimizer computation speed and memory usage

Below are the non-private classes in this module

'UInt8' (Unsigned 8-bit integer):
    - Range: 0 to 255
    - Size: 8 bits

'Int8' (Signed 8-bit integer):
    - Range: -128 to 127
    - Size: 8 bits

'UInt16' (Unsigned 16-bit integer):
    - Range: 0 to 65535
    - Size: 16 bits

'Int16' (Signed 16-bit integer):
    - Range: -32768 to 32767
    - Size: 16 bits

'UInt32' (Unsigned 32-bit integer):
    - Range: 0 to 4294967295
    - Size: 32 bits

'Int32' (Signed 32-bit integer):
    - Range: -2147483648 to 2147483647
    - Size: 32 bits

'UInt64' (Unsigned 64-bit integer):
    - Range: 0 to 18446744073709551615
    - Size: 64 bits

'Int64' (Signed 64-bit integer):
    - Range: -9223372036854775808 to 9223372036854775807
    - Size: 64 bits

'UInt128' (Unsigned 128-bit integer):
    - Range: 0 to 340282366920938463463374607431768211455
    - Size: 128 bits

'Int128' (Signed 128-bit integer):
    - Range: -170141183460469231731687303715884105728 to 170141183460469231731687303715884105727
    - Size: 128 bits

'UInt256' (Unsigned 256-bit integer):
    - Range: 0 to 115792089237316195423570985008687907853269984665640564039457584007913129639935
    - Size: 256 bits

'Int256' (Signed 256-bit integer):
    - Range: -57896044618658097711785492504343953926634992332820282019728792003956564819968 to 57896044618658097711785492504343953926634992332820282019728792003956564819967
    - Size: 256 bits

'Float8' (8-bit floating point number):
    - Range: -127 to 127
    - Size: 8 bits

'Float16' (16-bit floating point number):
    - Range: -65504 to 65504
    - Size: 16 bits

'Float32' (32-bit floating point number):
    - Range: -3.4028235e+38 to 3.4028235e+38
    - Size: 32 bits

'Float64' (64-bit floating point number):
    - Range: -1.7976931348623157e+308 to 1.7976931348623157e+308
    - Size: 64 bits

'Float128' (128-bit floating point number):
    - Range: -1.7976931348623157e+308 to 1.7976931348623157e+308
    - Size: 128 bits

'Float256' (256-bit floating point number):
    - Range: -1.7976931348623157e+308 to 1.7976931348623157e+308
    - Size: 256 bits

'Complex64' (64-bit complex number):
    - Range: -3.4028235e+38 to 3.4028235e+38
    - Size: 64 bits

'Complex128' (128-bit complex number):
    - Range: -1.7976931348623157e+308 to 1.7976931348623157e+308
    - Size: 128 bits

'Complex256' (256-bit complex number):
    - Range: -1.7976931348623157e+308 to 1.7976931348623157e+308
    - Size: 256 bits

Notes
-----

All overflows are handled by 'wrapping' the required value within the range using the modulo operator.
"""

class _FixedDataType:
    def __init__(self, value):
        self.value = self._validate(value)

    def _validate(self, value):
        raise NotImplementedError("Subclasses must implement this method")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

    # Basic arithmetic operations with wrapping
    def __add__(self, other):
        if isinstance(other, _FixedDataType):
            result = self.value + other.value
        else:
            result = self.value + other
        return self._wrap(result)

    def __sub__(self, other):
        if isinstance(other, _FixedDataType):
            result = self.value - other.value
        else:
            result = self.value - other
        return self._wrap(result)

    def __mul__(self, other):
        if isinstance(other, _FixedDataType):
            result = self.value * other.value
        else:
            result = self.value * other
        return self._wrap(result)

    def __floordiv__(self, other):
        if isinstance(other, _FixedDataType):
            result = self.value // other.value
        else:
            result = self.value // other
        return self._wrap(result)

    def __mod__(self, other):
        if isinstance(other, _FixedDataType):
            result = self.value % other.value
        else:
            result = self.value % other
        return self._wrap(result)

    def _wrap(self, result):
        """Ensure the result fits within the valid range and wrap if necessary."""
        raise NotImplementedError("Subclasses must implement this method")

class UInt8(_FixedDataType):
    def _validate(self, value):
        if not (0 <= value <= 255):
            raise ValueError(f"Value {value} is out of range for UInt8 (0-255)")
        return int(value)

    def _wrap(self, result):
        return UInt8(result % 256)

class Int8(_FixedDataType):
    def _validate(self, value):
        if not (-128 <= value <= 127):
            raise ValueError(f"Value {value} is out of range for Int8 (-128-127)")
        return int(value)

    def _wrap(self, result):
        return Int8((result + 128) % 256 - 128)

class UInt16(_FixedDataType):
    def _validate(self, value):
        if not (0 <= value <= 65535):
            raise ValueError(f"Value {value} is out of range for UInt16 (0-65535)")
        return int(value)

    def _wrap(self, result):
        return UInt16(result % 65536)

class Int16(_FixedDataType):
    def _validate(self, value):
        if not (-32768 <= value <= 32767):
            raise ValueError(f"Value {value} is out of range for Int16 (-32768-32767)")
        return int(value)

    def _wrap(self, result):
        return Int16((result + 32768) % 65536 - 32768)

class UInt32(_FixedDataType):
    def _validate(self, value):
        if not (0 <= value <= 4294967295):
            raise ValueError(f"Value {value} is out of range for UInt32 (0-4294967295)")
        return int(value)

    def _wrap(self, result):
        return UInt32(result % 4294967296)

class Int32(_FixedDataType):
    def _validate(self, value):
        if not (-2147483648 <= value <= 2147483647):
            raise ValueError(f"Value {value} is out of range for Int32 (-2147483648-2147483647)")
        return int(value)

    def _wrap(self, result):
        return Int32((result + 2147483648) % 4294967296 - 2147483648)

class UInt64(_FixedDataType):
    def _validate(self, value):
        if not (0 <= value <= 18446744073709551615):
            raise ValueError(f"Value {value} is out of range for UInt64 (0-18446744073709551615)")
        return int(value)

    def _wrap(self, result):
        return UInt64(result % 18446744073709551616)

class Int64(_FixedDataType):
    def _validate(self, value):
        if not (-9223372036854775808 <= value <= 9223372036854775807):
            raise ValueError(f"Value {value} is out of range for Int64 (-9223372036854775808-9223372036854775807)")
        return int(value)

    def _wrap(self, result):
        return Int64((result + 9223372036854775808) % 18446744073709551616 - 9223372036854775808)

class UInt128(_FixedDataType):
    def _validate(self, value):
        if not (0 <= value <= 340282366920938463463374607431768211455):
            raise ValueError(f"Value {value} is out of range for UInt128 (0-340282366920938463463374607431768211455)")
        return int(value)

    def _wrap(self, result):
        return UInt128(result % 340282366920938463463374607431768211456)

class Int128(_FixedDataType):
    def _validate(self, value):
        if not (-170141183460469231731687303715884105728 <= value <= 170141183460469231731687303715884105727):
            raise ValueError(f"Value {value} is out of range for Int128 (-170141183460469231731687303715884105728-170141183460469231731687303715884105727)")
        return int(value)

    def _wrap(self, result):
        return Int128((result + 170141183460469231731687303715884105728) % 340282366920938463463374607431768211456 - 170141183460469231731687303715884105728)

class UInt256(_FixedDataType):
    def _validate(self, value):
        if not (0 <= value <= 115792089237316195423570985008687907853269984665640564039457584007913129639935):
            raise ValueError(f"Value {value} is out of range for UInt256")
        return int(value)

    def _wrap(self, result):
        return UInt256(result % 115792089237316195423570985008687907853269984665640564039457584007913129639936)

class Int256(_FixedDataType):
    def _validate(self, value):
        if not (-57896044618658097711785492504343953926634992332820282019728792003956564819968 <= value <= 57896044618658097711785492504343953926634992332820282019728792003956564819967):
            raise ValueError(f"Value {value} is out of range for Int256")
        return int(value)

    def _wrap(self, result):
        return Int256((result + 57896044618658097711785492504343953926634992332820282019728792003956564819968) % 115792089237316195423570985008687907853269984665640564039457584007913129639936 - 57896044618658097711785492504343953926634992332820282019728792003956564819968)

class Float8(_FixedDataType):
    def _validate(self, value):
        if not (-127 <= value <= 127):
            raise ValueError(f"Value {value} is out of range for Float8 (-127-127)")
        return float(value)

    def _wrap(self, result):
        return Float8(result % 256 - 128)

class Float16(_FixedDataType):
    def _validate(self, value):
        if not (-65504 <= value <= 65504):
            raise ValueError(f"Value {value} is out of range for Float16 (-65504-65504)")
        return float(value)

    def _wrap(self, result):
        return Float16(result % 65536 - 32768)

class Float32(_FixedDataType):
    def _validate(self, value):
        if not (-3.4028235e+38 <= value <= 3.4028235e+38):
            raise ValueError(f"Value {value} is out of range for Float32 (-3.4028235e+38-3.4028235e+38)")
        return float(value)

    def _wrap(self, result):
        return Float32(result % 4294967296 - 2147483648)

class Float64(_FixedDataType):
    def _validate(self, value):
        if not (-1.7976931348623157e+308 <= value <= 1.7976931348623157e+308):
            raise ValueError(f"Value {value} is out of range for Float64 (-1.7976931348623157e+308-1.7976931348623157e+308)")
        return float(value)

    def _wrap(self, result):
        return Float64(result % 8589934592 - 4294967296)

class Float128(_FixedDataType):
    def _validate(self, value):
        if not (-1.7976931348623157e+308 <= value <= 1.7976931348623157e+308):
            raise ValueError(f"Value {value} is out of range for Float128 (-1.7976931348623157e+308-1.7976931348623157e+308)")
        return float(value)

    def _wrap(self, result):
        return Float128(result % 17179869184 - 8589934592)

class Float256(_FixedDataType):
    def _validate(self, value):
        if not (-1.7976931348623157e+308 <= value <= 1.7976931348623157e+308):
            raise ValueError(f"Value {value} is out of range for Float256 (-1.7976931348623157e+308-1.7976931348623157e+308)")
        return float(value)

    def _wrap(self, result):
        return Float256(result % 34359738368 - 17179869184)

class Complex64(_FixedDataType):
    def _validate(self, value):
        if not (-3.4028235e+38 <= value.real <= 3.4028235e+38):
            raise ValueError(f"Real part {value.real} is out of range for Complex64 (-3.4028235e+38-3.4028235e+38)")
        if not (-3.4028235e+38 <= value.imag <= 3.4028235e+38):
            raise ValueError(f"Imaginary part {value.imag} is out of range for Complex64 (-3.4028235e+38-3.4028235e+38)")
        return complex(value)

    def _wrap(self, result):
        real = result.real % 256 - 128
        imag = result.imag % 256 - 128
        return Complex64(complex(real, imag))

class Complex128(_FixedDataType):
    def _validate(self, value):
        if not (-1.7976931348623157e+308 <= value.real <= 1.7976931348623157e+308):
            raise ValueError(f"Real part {value.real} is out of range for Complex128 (-1.7976931348623157e+308-1.7976931348623157e+308)")
        if not (-1.7976931348623157e+308 <= value.imag <= 1.7976931348623157e+308):
            raise ValueError(f"Imaginary part {value.imag} is out of range for Complex128 (-1.7976931348623157e+308-1.7976931348623157e+308)")
        return complex(value)

    def _wrap(self, result):
        real = result.real % 65536 - 32768
        imag = result.imag % 65536 - 32768
        return Complex128(complex(real, imag))

class Complex256(_FixedDataType):
    def _validate(self, value):
        if not (-1.7976931348623157e+308 <= value.real <= 1.7976931348623157e+308):
            raise ValueError(f"Real part {value.real} is out of range for Complex256 (-1.7976931348623157e+308-1.7976931348623157e+308)")
        if not (-1.7976931348623157e+308 <= value.imag <= 1.7976931348623157e+308):
            raise ValueError(f"Imaginary part {value.imag} is out of range for Complex256 (-1.7976931348623157e+308-1.7976931348623157e+308)")
        return complex(value)

    def _wrap(self, result):
        real = result.real % 4294967296 - 2147483648
        imag = result.imag % 4294967296 - 2147483648
        return Complex256(complex(real, imag))