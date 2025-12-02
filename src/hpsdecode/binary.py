"""Binary reader for reading bits and common data types from a byte stream."""

__all__ = ["BinaryReader"]

import io
import struct


class BinaryReader:
    """A binary reader for reading bits, with convenient methods for common integer and floating-point types.

    .. code-block:: python

        reader = BinaryReader(b"\\x01\\x02\\x03\\x04")
        value = reader.read_uint32()  # Little-endian: 0x04030201

    """

    _bit_buffer: int
    _bit_count: int
    _stream: io.BytesIO

    def __init__(self, data: bytes) -> None:
        """Initialize the binary reader.

        :param data: The binary data to read from.
        """
        self._stream = io.BytesIO(data)
        self._bit_buffer = 0
        self._bit_count = 0

    @property
    def position(self) -> int:
        """Get the current byte position in the stream.

        :return: The current byte position.
        """
        byte_pos = self._stream.tell()
        if self._bit_count > 0:
            byte_pos -= 1

        return byte_pos

    def align_to_byte(self) -> None:
        """Align the read position to the next byte boundary.

        Discards any remaining bits in the current bit buffer.
        """
        self._bit_buffer = 0
        self._bit_count = 0

    def is_eof(self) -> bool:
        """Check if the end of the stream has been reached.

        :return: Whether the end of the stream is reached.
        """
        current_pos = self._stream.tell()
        self._stream.seek(0, io.SEEK_END)
        end_pos = self._stream.tell()
        self._stream.seek(current_pos)
        return current_pos >= end_pos and self._bit_count == 0

    def read_bits(self, n: int) -> int:
        """Read n bits from the stream.

        :param n: The number of bits to read (1-32).
        :return: The integer value of the bits read.
        """
        if n < 1 or n > 32:
            raise ValueError("Number of bits must be between 1 and 32")

        result = 0
        bits_remaining = n

        while bits_remaining > 0:
            if self._bit_count == 0:
                byte = self._stream.read(1)
                if not byte:
                    raise EOFError("Unexpected end of stream")

                self._bit_buffer = byte[0]
                self._bit_count = 8

            bits_to_read = min(bits_remaining, self._bit_count)
            mask = (1 << bits_to_read) - 1
            bits = (self._bit_buffer >> (self._bit_count - bits_to_read)) & mask

            result = (result << bits_to_read) | bits
            self._bit_count -= bits_to_read
            bits_remaining -= bits_to_read

        return result

    def read_bytes(self, n: int) -> bytes:
        """Read n bytes from the stream.

        Automatically aligns to byte boundary before reading.

        :param n: The number of bytes to read.
        :return: The bytes read from the stream.
        """
        self.align_to_byte()

        data = self._stream.read(n)
        if len(data) != n:
            raise EOFError(f"Expected {n} bytes, got {len(data)}")

        return data

    def read_uint8(self) -> int:
        """Read an unsigned 8-bit integer.

        :return: The integer value (between 0 and 255).
        """
        return self.read_bytes(1)[0]

    def read_uint16(self) -> int:
        """Read an unsigned 16-bit integer (little-endian).

        :return: The integer value (between 0 and 65535).
        """
        return int.from_bytes(self.read_bytes(2), byteorder="little", signed=False)

    def read_uint32(self) -> int:
        """Read an unsigned 32-bit integer (little-endian).

        :return: The integer value (between 0 and 4294967295).
        """
        return int.from_bytes(self.read_bytes(4), byteorder="little", signed=False)

    def read_int16(self) -> int:
        """Read a signed 16-bit integer (little-endian).

        :return: The integer value (between -32768 and 32767).
        """
        return int.from_bytes(self.read_bytes(2), byteorder="little", signed=True)

    def read_int32(self) -> int:
        """Read a signed 32-bit integer (little-endian).

        :return: The integer value (between -2147483648 and 2147483647).
        """
        return int.from_bytes(self.read_bytes(4), byteorder="little", signed=True)

    def read_float32(self) -> float:
        """Read a 32-bit floating point number (little-endian).

        :return: The floating point value.
        """
        return struct.unpack("<f", self.read_bytes(4))[0]
