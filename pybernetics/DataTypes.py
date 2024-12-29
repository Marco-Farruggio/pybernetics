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

All intiger overflows are handled by 'wrapping' the required value within the range using the modulo operator.
All complex overflows are handled by clamping the value to the maximum or minimum value.
"""

class _FixedIntigerDataType:
    _min = None
    _max = None

    def __init__(self, value):
        self.value = self._validate(value)

    def _validate(self, value):
        if self._min is None or self._max is None:
            raise NotImplementedError("Subclasses must define _min and _max")
        
        if not (self._min <= value <= self._max):
            raise ValueError(f"Value {value} is out of range for {self.__class__.__name__} ({self._min}-{self._max})")
        
        return int(value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"
    
    def __str__(self):
        return str(self.value)

    # Basic arithmetic operations with wrapping
    def __add__(self, other):
        if isinstance(other, _FixedIntigerDataType):
            result = self.value + other.value
        else:
            result = self.value + other
        return self._wrap(result)

    def __sub__(self, other):
        if isinstance(other, _FixedIntigerDataType):
            result = self.value - other.value
        else:
            result = self.value - other
        return self._wrap(result)

    def __mul__(self, other):
        if isinstance(other, _FixedIntigerDataType):
            result = self.value * other.value
        else:
            result = self.value * other
        return self._wrap(result)

    def __floordiv__(self, other):
        if isinstance(other, _FixedIntigerDataType):
            result = self.value // other.value
        else:
            result = self.value // other
        return self._wrap(result)

    def __mod__(self, other):
        if isinstance(other, _FixedIntigerDataType):
            result = self.value % other.value
        else:
            result = self.value % other
        return self._wrap(result)

    def _wrap(self, result):
        """Ensure the result fits within the valid range and wrap if necessary."""
        if self._min is None or self._max is None:
            raise NotImplementedError("Subclasses must define _min and _max for wrapping")

        # Check if the type is unsigned (min is 0)
        if self._min == 0:
            # Unsigned types: wrap between 0 and 2^n - 1
            return self.__class__(result % (self._max + 1))
        else:
            # Signed types: wrap between -2^n and 2^n-1 (for signed integer)
            range_size = self._max - self._min + 1
            return self.__class__((result - self._min) % range_size + self._min)

class _FixedFloatDataType:
    _min = None
    _max = None

    def __init__(self, value):
        self.value = self._validate(value)

    def _validate(self, value):
        """Validate that the value is within the valid range."""
        if self._min is None or self._max is None:
            raise NotImplementedError("Subclasses must define _min and _max")
        
        if not (self._min <= value <= self._max):
            raise ValueError(f"Value {value} is out of range for {self.__class__.__name__} ({self._min}-{self._max})")
        
        return float(value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"
    
    def __str__(self):
        return str(self.value)

    # Basic arithmetic operations
    def __add__(self, other):
        if isinstance(other, _FixedFloatDataType):
            result = self.value + other.value
        else:
            result = self.value + other
        return self._clamp(result)

    def __sub__(self, other):
        if isinstance(other, _FixedFloatDataType):
            result = self.value - other.value
        else:
            result = self.value - other
        return self._clamp(result)

    def __mul__(self, other):
        if isinstance(other, _FixedFloatDataType):
            result = self.value * other.value
        else:
            result = self.value * other
        return self._clamp(result)

    def __truediv__(self, other):
        if isinstance(other, _FixedFloatDataType):
            result = self.value / other.value
        else:
            result = self.value / other
        return self._clamp(result)

    def __floordiv__(self, other):
        if isinstance(other, _FixedFloatDataType):
            result = self.value // other.value
        else:
            result = self.value // other
        return self._clamp(result)

    def __mod__(self, other):
        if isinstance(other, _FixedFloatDataType):
            result = self.value % other.value
        else:
            result = self.value % other
        return self._clamp(result)

    def _clamp(self, result):
        """Clamp the result according to the valid range."""
        if self._min is None or self._max is None:
            raise NotImplementedError("Subclasses must define _min and _max for clamping")

        # Clamp the result to the valid range for floating-point types
        return self.__class__(max(self._min, min(result, self._max)))

class _FixedComplexDataType:
    _min = None
    _max = None

    def __init__(self, value):
        self.value = self._validate(value)

    def _validate(self, value):
        """Validates the complex number's real and imaginary parts."""
        if self._min is None or self._max is None:
            raise NotImplementedError("Subclasses must define _min and _max")
        
        # Validate real part
        if not (self._min <= value.real <= self._max):
            raise ValueError(f"Real part {value.real} is out of range for {self.__class__.__name__} ({self._min}-{self._max})")
        
        # Validate imaginary part
        if not (self._min <= value.imag <= self._max):
            raise ValueError(f"Imaginary part {value.imag} is out of range for {self.__class__.__name__} ({self._min}-{self._max})")
        
        return complex(value)

    def _clamp(self, result):
        """Clamps the result's real and imaginary parts to fit within the valid range."""
        
        real = max(self._min, min(result.real, self._max))
        imag = max(self._min, min(result.imag, self._max))
        
        return self.__class__(complex(real, imag))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"
    
    def __str__(self):
        return str(self.value)

    # Basic arithmetic operations
    def __add__(self, other):
        if isinstance(other, _FixedComplexDataType):
            result = self.value + other.value
        else:
            result = self.value + other
        return self._clamp(result)

    def __sub__(self, other):
        if isinstance(other, _FixedComplexDataType):
            result = self.value - other.value
        else:
            result = self.value - other
        return self._clamp(result)

    def __mul__(self, other):
        if isinstance(other, _FixedComplexDataType):
            result = self.value * other.value
        else:
            result = self.value * other
        return self._clamp(result)

    def __truediv__(self, other):
        if isinstance(other, _FixedComplexDataType):
            result = self.value / other.value
        else:
            result = self.value / other
        return self._clamp(result)

    def __floordiv__(self, other):
        if isinstance(other, _FixedComplexDataType):
            result = self.value // other.value
        else:
            result = self.value // other
        return self._clamp(result)

    def __mod__(self, other):
        if isinstance(other, _FixedComplexDataType):
            result = self.value % other.value
        else:
            result = self.value % other
        return self._clamp(result)

    def __eq__(self, other):
        if isinstance(other, _FixedComplexDataType):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __abs__(self):
        return abs(self.value)

    def __neg__(self):
        return self._clamp(-self.value)

    def __pos__(self):
        return self._clamp(+self.value)

    def __pow__(self, other):
        if isinstance(other, _FixedComplexDataType):
            result = self.value ** other.value
        else:
            result = self.value ** other
        return self._clamp(result)

class UInt8(_FixedIntigerDataType):
    _min = 0
    _max = 255

class Int8(_FixedIntigerDataType):
    _min = -128
    _max = 127

class UInt16(_FixedIntigerDataType):
    _min = 0
    _max = 65535

class Int16(_FixedIntigerDataType):
    _min = -32768
    _max = 32767

class UInt32(_FixedIntigerDataType):
    _min = 0
    _max = 4294967295

class Int32(_FixedIntigerDataType):
    _min = -2147483648
    _max = 2147483647

class UInt64(_FixedIntigerDataType):
    _min = 0
    _max = 18446744073709551615

class Int64(_FixedIntigerDataType):
    _min = -9223372036854775808
    _max = 9223372036854775807

class UInt128(_FixedIntigerDataType):
    _min = 0
    _max = 340282366920938463463374607431768211455

class Int128(_FixedIntigerDataType):
    _min = -170141183460469231731687303715884105728
    _max = 170141183460469231731687303715884105727

class UInt256(_FixedIntigerDataType):
    _min = 0
    _max = 115792089237316195423570985008687907853269984665640564039457584007913129639935

class Int256(_FixedIntigerDataType):
    _min = -57896044618658097711785492504343953926634992332820282019728792003956564819968
    _max = 57896044618658097711785492504343953926634992332820282019728792003956564819967

class Float8(_FixedFloatDataType):
    _min = -127
    _max = 127

class Float16(_FixedFloatDataType):
    _min = -65504
    _max = 65504

class Float32(_FixedFloatDataType):
    _min = -3.4028235e+38
    _max = 3.4028235e+38

class Float64(_FixedFloatDataType):
    _min = -1.7976931348623157e+308
    _max = 1.7976931348623157e+308

class Float128(_FixedFloatDataType):
    _min = -1.7976931348623157e+308
    _max = 1.7976931348623157e+308

class Float256(_FixedFloatDataType):
    _min = -1.7976931348623157e+308
    _max = 1.7976931348623157e+308

class Complex64(_FixedComplexDataType):
    _min = -3.4028235e+38
    _max = 3.4028235e+38

class Complex128(_FixedComplexDataType):
    _min = -1.7976931348623157e+308
    _max = 1.7976931348623157e+308

class Complex256(_FixedComplexDataType):
    _min = -1.7976931348623157e+308
    _max = 1.7976931348623157e+308
