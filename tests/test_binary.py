import pytest

from hpsdecode.binary import BinaryReader


class TestBinaryReaderUnsignedIntegers:
    """Tests for reading unsigned integer types."""

    def test_read_uint8(self) -> None:
        """Read unsigned 8-bit integers."""
        data = b"\x00\x01\xff\x7f\x80"
        reader = BinaryReader(data)

        assert reader.read_uint8() == 0x00
        assert reader.read_uint8() == 0x01
        assert reader.read_uint8() == 0xFF
        assert reader.read_uint8() == 0x7F
        assert reader.read_uint8() == 0x80

    def test_read_uint16_little_endian(self) -> None:
        """Read unsigned 16-bit integers in little-endian order."""
        data = b"\x00\x00\x01\x00\xff\xff\x34\x12"
        reader = BinaryReader(data)

        assert reader.read_uint16() == 0x0000
        assert reader.read_uint16() == 0x0001
        assert reader.read_uint16() == 0xFFFF
        assert reader.read_uint16() == 0x1234

    def test_read_uint32_little_endian(self) -> None:
        """Read unsigned 32-bit integers in little-endian order."""
        data = b"\x00\x00\x00\x00\x01\x00\x00\x00\xff\xff\xff\xff\x78\x56\x34\x12"
        reader = BinaryReader(data)

        assert reader.read_uint32() == 0x00000000
        assert reader.read_uint32() == 0x00000001
        assert reader.read_uint32() == 0xFFFFFFFF
        assert reader.read_uint32() == 0x12345678


class TestBinaryReaderSignedIntegers:
    """Tests for reading signed integer types."""

    def test_read_int16_signed_values(self) -> None:
        """Read signed 16-bit integers including negative values."""
        data = b"\x00\x00\x01\x00\xff\xff\x00\x80"
        reader = BinaryReader(data)

        assert reader.read_int16() == 0
        assert reader.read_int16() == 1
        assert reader.read_int16() == -1
        assert reader.read_int16() == -32768

    def test_read_int32_signed_values(self) -> None:
        """Read signed 32-bit integers including negative values."""
        data = b"\x00\x00\x00\x00\x01\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x80"
        reader = BinaryReader(data)

        assert reader.read_int32() == 0
        assert reader.read_int32() == 1
        assert reader.read_int32() == -1
        assert reader.read_int32() == -2147483648


class TestBinaryReaderFloatingPoint:
    """Tests for reading floating-point types."""

    def test_read_float32_ieee754(self) -> None:
        """Read IEEE 754 single-precision floats."""
        data = b"\x00\x00\x00\x00\x00\x00\x80\x3f\x00\x00\x00\xc0"
        reader = BinaryReader(data)

        assert reader.read_float32() == 0.0
        assert reader.read_float32() == pytest.approx(1.0, abs=1e-6)
        assert reader.read_float32() == pytest.approx(-2.0, abs=1e-6)


class TestBinaryReaderBitOperations:
    """Tests for bit-level reading operations."""

    def test_read_bits_single_byte(self) -> None:
        """Read bits from within a single byte."""
        data = b"\xf0"
        reader = BinaryReader(data)

        assert reader.read_bits(4) == 0xF
        assert reader.read_bits(4) == 0x0

    def test_read_bits_across_boundaries(self) -> None:
        """Read bits spanning multiple bytes."""
        data = b"\xaa\x55"
        reader = BinaryReader(data)

        assert reader.read_bits(4) == 0xA
        assert reader.read_bits(4) == 0xA
        assert reader.read_bits(4) == 0x5
        assert reader.read_bits(4) == 0x5

    def test_read_bits_variable_sizes(self) -> None:
        """Read different bit counts in sequence."""
        data = b"\xff\x00\x81"
        reader = BinaryReader(data)

        assert reader.read_bits(8) == 0xFF
        assert reader.read_bits(1) == 0
        assert reader.read_bits(7) == 0
        assert reader.read_bits(1) == 1
        assert reader.read_bits(7) == 1

    def test_read_bits_beyond_eof(self) -> None:
        """read_bits raises EOFError at stream end."""
        data = b"\x01"
        reader = BinaryReader(data)

        reader.read_bits(8)

        with pytest.raises(EOFError, match="Unexpected end of stream"):
            reader.read_bits(1)

    def test_read_bits_invalid_count(self) -> None:
        """read_bits validates bit range."""
        reader = BinaryReader(b"\x01")

        with pytest.raises(ValueError, match="between 1 and 32"):
            reader.read_bits(0)

        with pytest.raises(ValueError, match="between 1 and 32"):
            reader.read_bits(33)


class TestBinaryReaderByteOperations:
    """Tests for byte-level reading operations."""

    def test_read_bytes_aligns(self) -> None:
        """read_bytes automatically aligns to byte boundary."""
        data = b"\xaa\x55\xff\x00"
        reader = BinaryReader(data)

        reader.read_bits(4)
        result = reader.read_bytes(2)

        assert result == b"\x55\xff"

    def test_read_bytes_insufficient_data(self) -> None:
        """read_bytes raises EOFError when not enough data."""
        data = b"\x01\x02"
        reader = BinaryReader(data)

        with pytest.raises(EOFError, match="Expected 3 bytes"):
            reader.read_bytes(3)


class TestBinaryReaderPositionTracking:
    """Tests for stream position tracking."""

    def test_align_to_byte(self) -> None:
        """Align to next byte boundary after partial read."""
        data = b"\xaa\x55\xff"
        reader = BinaryReader(data)

        reader.read_bits(4)
        reader.align_to_byte()

        assert reader.read_uint8() == 0x55

    def test_position_tracks_bytes_read(self) -> None:
        """Position property tracks byte-aligned reads."""
        data = b"\x01\x02\x03\x04"
        reader = BinaryReader(data)

        assert reader.position == 0
        reader.read_uint8()
        assert reader.position == 1
        reader.read_uint16()
        assert reader.position == 3

    def test_position_with_partial_bits(self) -> None:
        """Position stays at byte boundary until full byte consumed."""
        data = b"\xaa\x55\xff"
        reader = BinaryReader(data)

        reader.read_bits(4)
        assert reader.position == 0
        reader.read_bits(4)
        assert reader.position == 1

    def test_is_eof(self) -> None:
        """Detect end of stream."""
        data = b"\x01\x02"
        reader = BinaryReader(data)

        assert not reader.is_eof()
        reader.read_uint8()
        assert not reader.is_eof()
        reader.read_uint8()
        assert reader.is_eof()
