"""Mesh data structures for decoded HPS content."""

from __future__ import annotations

__all__ = ["Edge", "HPSMesh", "HPSPackedScan", "SchemaType", "Spline"]

import dataclasses
import typing as t

from hpsdecode.export import ExportFormat, export_mesh

if t.TYPE_CHECKING:
    import os

    import numpy as np
    import numpy.typing as npt

    from hpsdecode.commands import AnyFaceCommand, AnyVertexCommand
    from hpsdecode.export.obj import MaterialConfig


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
class Spline:
    """A 3D spline defined by control points."""

    #: The name/identifier of the spline.
    name: str

    #: The control points of the spline as an (N, 3) float array.
    control_points: npt.NDArray[np.floating]

    #: The radius of the spline.
    radius: float

    #: Whether the spline is cyclic (i.e., forms a closed loop).
    is_cyclic: bool

    #: Metadata or flags associated with the spline.
    misc: int

    @property
    def num_control_points(self) -> int:
        """The number of control points in the spline."""
        return int(self.control_points.shape[0])


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

    #: The default vertex color to use if it cannot be determined from the data.
    default_vertex_color: int | None

    #: The default face color to use if it cannot be determined from the data.
    default_face_color: int | None

    #: The vertex color data.
    vertex_colors_data: bytes | None

    #: The texture coordinate data.
    texture_coords_data: bytes | None

    #: A list of texture images.
    texture_images: list[bytes]

    #: A list of splines associated with the scan, if any.
    splines: list[Spline]

    #: Parsed vertex command sequence.
    vertex_commands: list[AnyVertexCommand]

    #: Parsed face command sequence.
    face_commands: list[AnyFaceCommand]

    #: The value used for integrity checking, if available.
    check_value: int | None

    #: The HPS file properties (e.g., encryption keys, package locks).
    properties: dict[str, str]

    @property
    def is_encrypted(self) -> bool:
        """Whether the scan data is encrypted."""
        return self.schema == "CE"


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

    #: Texture coordinates as (M, 2) float array, or empty.
    uv: npt.NDArray[np.floating]

    #: Texture image data (raw bytes, multiple images possible).
    texture_images: list[bytes] = dataclasses.field(default_factory=list)

    #: The splines associated with the mesh, if any.
    splines: list[Spline] = dataclasses.field(default_factory=list)

    @property
    def num_faces(self) -> int:
        """Number of faces in the mesh."""
        return int(self.faces.shape[0])

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return int(self.vertices.shape[0])

    @property
    def has_texture_coords(self) -> bool:
        """Whether the mesh has texture coordinates."""
        return self.uv.size > 0

    @property
    def has_vertex_colors(self) -> bool:
        """Whether the mesh has per-vertex colors."""
        return self.vertex_colors.size > 0

    @property
    def has_face_colors(self) -> bool:
        """Whether the mesh has per-face colors."""
        return self.face_colors.size > 0

    @property
    def has_splines(self) -> bool:
        """Whether the mesh has any splines."""
        return len(self.splines) > 0

    @property
    def has_textures(self) -> bool:
        """Whether the mesh has texture image data."""
        return len(self.texture_images) > 0

    def export(
        self,
        output_path: str | os.PathLike[str],
        export_format: ExportFormat | None = None,
        *,
        binary: bool = True,
        include_colors: bool = True,
        include_textures: bool = True,
        material: MaterialConfig | None = None,
    ) -> None:
        """Export an HPS mesh to a file.

        :param output_path: The output file path.
        :param export_format: The export format. If None, inferred from file extension.
        :param binary: Whether to use binary format (STL/PLY only). Default is ``True``.
        :param include_colors: Whether to include vertex colors if available (OBJ/PLY only). Default is ``True``.
        :param include_textures: Whether to include textures if available (OBJ only)
            or bake textures to vertex colors (PLY only). Default is ``True``.
        :param material: The material configuration (OBJ only). If ``None``, default values are used.
        :raises ValueError: If the format is unsupported or incompatible with options.
        """
        export_mesh(
            self,
            output_path,
            export_format,
            binary=binary,
            include_colors=include_colors,
            include_textures=include_textures,
            material=material,
        )
