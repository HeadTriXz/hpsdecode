"""Texture coordinate decompression utilities."""

from __future__ import annotations

__all__ = ["decompress_texture_coord", "parse_texture_coords"]

import typing as t

import numpy as np

from hpsdecode.binary import BinaryReader

if t.TYPE_CHECKING:
    import numpy.typing as npt


#: Bit 15 indicates value is outside [0, 1].
_OUTSIDE_RANGE_BIT = 0x8000

#: Mask for the lower 15 bits representing the coordinate value.
_COORD_MASK = 0x7FFF

#: Scale factor for [0, 1] range.
_SCALE_INSIDE = 1.0 / 32767.0

#: Scale factor for [-256, 256] range.
_SCALE_OUTSIDE = 512.0 / 32767.0

#: Marker value indicating a missing texture coordinate.
_NO_UV_MARKER = 0xFFFFFFFF


def decompress_texture_coord(compressed: int) -> tuple[float, float]:
    """Decompress a single texture coordinate from 32-bit representation.

    HPS stores two 16-bit values (u, v) packed into a single 32-bit integer.
    Each 16-bit value uses:
        - Bit 15: Range flag (0 = [0, 1], 1 = [-256, 256]).
        - Bits 0-14: Coordinate value.

    :param compressed: The 32-bit compressed coordinate.
    :return: A tuple of (u, v) as floats.
    """
    u_bits = compressed & 0xFFFF
    u = _decompress_component(u_bits)

    v_bits = (compressed >> 16) & 0xFFFF
    v = _decompress_component(v_bits)

    return u, v


def _decompress_component(bits: int) -> float:
    """Decompress a single U or V component from 16-bit representation.

    :param bits: The 16-bit value to decompress.
    :return: The decompressed float value.
    """
    is_outside_range = (bits & _OUTSIDE_RANGE_BIT) != 0
    value = bits & _COORD_MASK

    if is_outside_range:
        # Map [0, 32767] to [-256, 256]
        return (value * _SCALE_OUTSIDE) - 256.0
    else:
        # Map [0, 32767] to [0, 1]
        return value * _SCALE_INSIDE


def parse_texture_coords(data: bytes, num_vertices: int) -> npt.NDArray[np.floating]:
    """Parse texture coordinates.

    The format stores one or more UV coordinates per vertex, with a flag byte
    indicating the storage mode:

        - Flag = 1: Single UV shared by all faces connected to this vertex.
        - Flag = 0xFF: Multiple UVs (exact count unknown without topology).
        - Other: Exact number of UVs for connected faces.

    :param data: The raw binary texture coordinate data.
    :param num_vertices: The expected number of vertices.
    :return: Array of UV coordinates with shape (num_vertices, 2).
    :raises ValueError: If data is malformed or insufficient.
    """
    reader = BinaryReader(data)
    uvs = []

    for vertex_idx in range(num_vertices):
        try:
            flag = reader.read_uint8()

            # Read the first UV coordinate for this vertex
            uv = _read_uv_coordinate(reader, vertex_idx)
            uvs.append(uv)

            # Skip additional UV coordinates if present
            num_additional = _count_additional_uvs(flag)
            for _ in range(num_additional):
                reader.read_uint32()

        except EOFError as e:
            raise ValueError(f"Unexpected end of texture data at vertex {vertex_idx}/{num_vertices}") from e

    if len(uvs) != num_vertices:
        raise ValueError(f"UV count mismatch: expected {num_vertices}, got {len(uvs)}")

    return np.array(uvs, dtype=np.float32)


def _read_uv_coordinate(reader: BinaryReader, vertex_idx: int) -> list[float]:
    """Read and decompress a single UV coordinate.

    :param reader: The binary reader.
    :param vertex_idx: The current vertex index (for error messages).
    :return: A [u, v] pair.
    :raises EOFError: If insufficient data.
    """
    try:
        compressed = reader.read_uint32()
    except EOFError as e:
        raise ValueError(f"Insufficient data for UV at vertex {vertex_idx}") from e

    if compressed == _NO_UV_MARKER:
        return [0.0, 0.0]

    u, v = decompress_texture_coord(compressed)
    return [u, v]


def _count_additional_uvs(flag: int) -> int:
    """Determine how many additional UV coordinates to skip after the first.

    :param flag: The flag byte from the data stream.
    :return: Number of additional UVs to skip (0 if only one UV).
    """
    if flag == 1 or flag == 0xFF:
        return 0

    return max(0, flag - 1)
