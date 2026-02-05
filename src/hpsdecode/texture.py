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


def parse_texture_coords(data: bytes, num_vertices: int, faces: npt.NDArray[np.integer]) -> npt.NDArray[np.floating]:
    """Parse texture coordinates.

    The format stores one or more UV coordinates per vertex, with a flag byte
    indicating the storage mode:

        - Flag = 1: Single UV shared by all faces connected to this vertex.
        - Flag = 0xFF: Multiple UVs (one per connected face).
        - Other: Exact number of UVs for connected faces.

    :param data: The raw binary texture coordinate data.
    :param num_vertices: The expected number of vertices.
    :param faces: Face indices array of shape (num_faces, 3).
    :return: Array of UV coordinates with shape (num_faces * 3, 2).
    :raises ValueError: If data is malformed or insufficient.
    """
    reader = BinaryReader(data)
    num_faces = faces.shape[0]

    face_corners = faces.ravel()
    vertex_corners: list[list[int]] = [[] for _ in range(num_vertices)]
    for corner_idx, vertex_idx in enumerate(face_corners):
        vertex_corners[vertex_idx].append(corner_idx)

    uvs = np.zeros((num_faces * 3, 2), dtype=np.float32)

    for vertex_idx in range(num_vertices):
        try:
            flag = reader.read_uint8()
            corners = vertex_corners[vertex_idx]

            if flag == 1:
                # Single UV shared by all corners
                compressed = reader.read_uint32()
                if compressed != _NO_UV_MARKER:
                    u, v = decompress_texture_coord(compressed)
                    for corner_idx in corners:
                        uvs[corner_idx] = [u, v]
            else:
                # Multiple UVs (one per face)
                if flag != 0xFF:
                    num_faces_for_vertex = len(corners)
                    if flag != num_faces_for_vertex:
                        raise ValueError(
                            f"Mismatch at vertex {vertex_idx}: flag={flag}, expected={num_faces_for_vertex}"
                        )

                corners_sorted = sorted(corners, key=lambda c: c // 3)
                for corner_idx in corners_sorted:
                    compressed = reader.read_uint32()
                    if compressed != _NO_UV_MARKER:
                        u, v = decompress_texture_coord(compressed)
                        uvs[corner_idx] = [u, v]

        except EOFError as e:
            raise ValueError(f"Unexpected end of texture data at vertex {vertex_idx}/{num_vertices}") from e

    return uvs
