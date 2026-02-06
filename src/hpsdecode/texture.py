"""Texture coordinate decompression utilities."""

from __future__ import annotations

__all__ = [
    "decompress_texture_coord",
    "parse_texture_coords",
    "face_colors_to_vertex_colors",
    "texture_to_vertex_colors",
    "deduplicate_vertices_for_uv",
]

import io
import logging
import typing as t

import numpy as np
from PIL import Image

from hpsdecode.binary import BinaryReader

if t.TYPE_CHECKING:
    import numpy.typing as npt

    from hpsdecode.mesh import HPSMesh

logger = logging.getLogger(__name__)


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


def face_colors_to_vertex_colors(mesh: HPSMesh) -> npt.NDArray[np.uint8]:
    """Convert face colors to vertex colors by averaging.

    :param mesh: The mesh containing face colors.
    :return: An array of vertex colors (N, 3).
    """
    vertex_colors = np.zeros((mesh.num_vertices, 3), dtype=np.float32)
    vertex_counts = np.zeros(mesh.num_vertices, dtype=np.int32)

    for face_idx, face in enumerate(mesh.faces):
        face_color = mesh.face_colors[face_idx].astype(np.float32)
        for vertex_idx in face:
            vertex_colors[vertex_idx] += face_color
            vertex_counts[vertex_idx] += 1

    mask = vertex_counts > 0
    vertex_colors[mask] /= vertex_counts[mask, np.newaxis]

    return vertex_colors.astype(np.uint8)


def texture_to_vertex_colors(mesh: HPSMesh) -> npt.NDArray[np.uint8]:
    """Convert texture coordinates to vertex colors by sampling the texture image.

    :param mesh: The mesh containing texture data and UV coordinates.
    :return: An array of vertex colors (N, 3).
    """
    if not mesh.has_textures:
        raise ValueError("Mesh has no texture images.")

    if len(mesh.texture_images) > 1:
        logger.warning("Multiple texture images found; using the first one only.")

    image = Image.open(io.BytesIO(mesh.texture_images[0]))
    if image.mode != "RGB":
        image = image.convert("RGB")

    r, g, b = image.split()
    image = Image.merge("RGB", (b, g, r))

    width, height = image.size
    image_array = np.array(image, dtype=np.uint8)

    num_faces = mesh.num_faces
    uv_coords = mesh.uv.reshape(num_faces, 3, 2)

    vertex_color_samples: dict[int, list[npt.NDArray[np.uint8]]] = {}
    for face_idx, face in enumerate(mesh.faces):
        for corner_idx in range(3):
            vertex_idx = int(face[corner_idx])
            u, v = uv_coords[face_idx, corner_idx]

            x = int(np.clip(u * (width - 1), 0, width - 1))
            y = int(np.clip(v * (height - 1), 0, height - 1))

            color = image_array[y, x]
            if vertex_idx not in vertex_color_samples:
                vertex_color_samples[vertex_idx] = []

            vertex_color_samples[vertex_idx].append(color)

    vertex_colors = np.zeros((mesh.num_vertices, 3), dtype=np.uint8)
    for vertex_idx in range(mesh.num_vertices):
        if vertex_idx in vertex_color_samples:
            samples = vertex_color_samples[vertex_idx]
            avg_color = np.mean(samples, axis=0).astype(np.uint8)
            vertex_colors[vertex_idx] = avg_color
        else:
            vertex_colors[vertex_idx] = [128, 128, 128]

    return vertex_colors


def deduplicate_vertices_for_uv(
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    uv_coords: npt.NDArray[np.floating],
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.integer],
]:
    """Create unique vertex and UV coordinate combinations.

    :param vertices: The vertices of the mesh (N, 3).
    :param faces: The face indices of the mesh (M, 3).
    :param uv_coords: The UV coordinates per face corner (M, 3, 2).
    :return: (new_vertices, new_uvs, new_faces) where indices align.
    """
    vertex_uv_map: dict[tuple[int, tuple[float, float]], int] = {}
    new_vertices: list[npt.NDArray[np.floating]] = []
    new_uvs: list[tuple[float, float]] = []
    new_faces: list[list[int]] = []

    for face_idx, face in enumerate(faces):
        new_face = []
        for corner_idx in range(3):
            vertex_idx = int(face[corner_idx])
            uv = tuple(float(x) for x in uv_coords[face_idx, corner_idx])

            key = (vertex_idx, uv)
            if key not in vertex_uv_map:
                unique_idx = len(new_vertices)
                vertex_uv_map[key] = unique_idx
                new_vertices.append(vertices[vertex_idx])
                new_uvs.append(uv)

            new_face.append(vertex_uv_map[key])

        new_faces.append(new_face)

    return (
        np.array(new_vertices, dtype=np.float32),
        np.array(new_uvs, dtype=np.float32),
        np.array(new_faces, dtype=np.int32),
    )
