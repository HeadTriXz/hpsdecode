"""PLY format exporter."""

from __future__ import annotations

__all__ = ["PLYExporter"]

import struct
import typing as t
from pathlib import Path

import numpy as np

from hpsdecode.export.base import BaseExporter
from hpsdecode.texture import face_colors_to_vertex_colors, texture_to_vertex_colors

if t.TYPE_CHECKING:
    import os

    import numpy.typing as npt

    from hpsdecode.mesh import HPSMesh


class PLYExporter(BaseExporter):
    """Export meshes to PLY format with optional vertex colors."""

    #: Whether to use binary PLY format. If ``False``, ASCII PLY will be used.
    binary: bool

    #: Whether to include vertex colors if available. Default is ``True``.
    include_colors: bool = True

    #: Whether to bake textures to vertex colors. Only applies if the mesh has textures and UVs. Default is ``True``.
    include_textures: bool

    def __init__(
        self,
        binary: bool = True,
        include_colors: bool = True,
        include_textures: bool = True,
    ) -> None:
        """Initialize the PLY exporter.

        :param binary: Whether to use binary PLY format. Default is ``True``.
        :param include_colors: Whether to include vertex colors if available. Default is ``True``.
        :param include_textures: Whether to bake textures to vertex colors. Default is ``True``.
        """
        self.binary = binary
        self.include_colors = include_colors
        self.include_textures = include_textures

    def export(self, mesh: HPSMesh, output_path: str | os.PathLike[str]) -> None:
        """Export a mesh to PLY format.

        :param mesh: The mesh to export.
        :param output_path: The output file path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        use_texture = mesh.has_textures and mesh.has_texture_coords and self.include_textures
        use_vertex_colors = mesh.has_vertex_colors and self.include_colors
        use_face_colors = mesh.has_face_colors and self.include_colors

        vertex_colors = None
        if use_texture:
            vertex_colors = texture_to_vertex_colors(mesh)
        elif use_vertex_colors:
            vertex_colors = mesh.vertex_colors
        elif use_face_colors:
            vertex_colors = face_colors_to_vertex_colors(mesh)

        if self.binary:
            self._export_binary(mesh, path, vertex_colors)
        else:
            self._export_ascii(mesh, path, vertex_colors)

    def _export_binary(
        self,
        mesh: HPSMesh,
        path: Path,
        vertex_colors: npt.NDArray[np.uint8] | None,
    ) -> None:
        """Export mesh to binary PLY format.

        :param mesh: The mesh to export.
        :param path: The output file path.
        :param vertex_colors: The vertex colors to include (optional).
        """
        has_colors = vertex_colors is not None and vertex_colors.size > 0

        with path.open("wb") as f:
            header = self._generate_header(mesh.num_vertices, mesh.num_faces, has_colors, binary=True)
            f.write(header.encode("ascii"))

            for i in range(mesh.num_vertices):
                v = mesh.vertices[i]
                f.write(struct.pack("<fff", v[0], v[1], v[2]))

                if has_colors:
                    color = vertex_colors[i]
                    f.write(struct.pack("<BBB", color[0], color[1], color[2]))

            for face in mesh.faces:
                f.write(struct.pack("<B", 3))
                f.write(struct.pack("<III", face[0], face[1], face[2]))

    def _export_ascii(
        self,
        mesh: HPSMesh,
        path: Path,
        vertex_colors: npt.NDArray[np.uint8] | None,
    ) -> None:
        """Export mesh to ASCII PLY format.

        :param mesh: The mesh to export.
        :param path: The output file path.
        :param vertex_colors: The vertex colors to include (optional).
        """
        has_colors = vertex_colors is not None and vertex_colors.size > 0

        with path.open("w", encoding="ascii") as f:
            header = self._generate_header(mesh.num_vertices, mesh.num_faces, has_colors, binary=False)
            f.write(header)

            for i in range(mesh.num_vertices):
                v = mesh.vertices[i]
                if has_colors:
                    color = vertex_colors[i]
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
                else:
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            for face in mesh.faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    @staticmethod
    def _generate_header(num_vertices: int, num_faces: int, has_colors: bool, binary: bool) -> str:
        """Generate PLY header.

        :param num_vertices: Number of vertices in the mesh.
        :param num_faces: Number of faces in the mesh.
        :param has_colors: Whether vertex colors are included.
        :param binary: Whether the format is binary or ASCII.
        :return: The PLY header as a string.
        """
        lines = [
            "ply",
            f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
            "comment hpsdecode",
            f"element vertex {num_vertices}",
            "property float x",
            "property float y",
            "property float z",
        ]

        if has_colors:
            lines.extend([
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ])

        lines.extend([
            f"element face {num_faces}",
            "property list uchar int vertex_indices",
            "end_header",
        ])

        return "\n".join(lines) + "\n"
