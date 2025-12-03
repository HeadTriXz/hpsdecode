"""Mesh data structures for decoded HPS content."""

from __future__ import annotations

__all__ = ["Edge", "HPSMesh", "HPSPackedScan", "SchemaType"]

import dataclasses
import typing as t

if t.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from hpsdecode.commands import AnyFaceCommand, AnyVertexCommand


SchemaType: t.TypeAlias = t.Literal["CA", "CB", "CC", "CE"]


@dataclasses.dataclass
class Edge:
    """An edge connecting two vertex indices."""

    #: The starting vertex index.
    start: int

    #: The ending vertex index.
    end: int

    def __repr__(self) -> str:
        """The string representation of the edge."""
        return f"({self.start} → {self.end})"


@dataclasses.dataclass
class HPSPackedScan:
    """Metadata and commands from a packed HPS scan."""

    #: The compression schema identifier.
    schema: SchemaType

    #: Expected vertex count from file metadata.
    num_vertices: int

    #: Expected face count from file metadata.
    num_faces: int

    #: The raw vertex data.
    vertex_data: bytes

    #: The raw face data.
    face_data: bytes

    #: Parsed vertex command sequence.
    vertex_commands: list[AnyVertexCommand]

    #: Parsed face command sequence.
    face_commands: list[AnyFaceCommand]


@dataclasses.dataclass
class HPSMesh:
    """Decoded 3D mesh data."""

    #: Vertex positions as (N, 3) float array.
    vertices: npt.NDArray[np.floating]

    #: Face indices as (M, 3) integer array.
    faces: npt.NDArray[np.integer]

    #: Per-vertex RGB colors as (N, 3) uint8 array, or empty.
    vertex_colors: npt.NDArray[np.uint8]

    #: Per-face RGB colors as (M, 3) uint8 array, or empty.
    face_colors: npt.NDArray[np.uint8]

    #: Texture coordinates as (N, 2) float array, or empty.
    uv: npt.NDArray[np.floating]

    @property
    def num_faces(self) -> int:
        """Number of faces in the mesh."""
        return int(self.faces.shape[0])

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return int(self.vertices.shape[0])
