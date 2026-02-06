"""STL format exporter."""

from __future__ import annotations

__all__ = ["STLExporter"]

import struct
import typing as t
from pathlib import Path

import numpy as np

from hpsdecode.export.base import BaseExporter

if t.TYPE_CHECKING:
    import os

    import numpy.typing as npt

    from hpsdecode.mesh import HPSMesh


class STLExporter(BaseExporter):
    """Export meshes to STL format (geometry only)."""

    #: Whether to use binary STL format. If ``False``, ASCII STL will be used.
    binary: bool

    def __init__(self, binary: bool = True) -> None:
        """Initialize the STL exporter.

        :param binary: Whether to use binary STL format. Default is ``True``.
        """
        self.binary = binary

    def export(self, mesh: HPSMesh, output_path: str | os.PathLike[str]) -> None:
        """Export a mesh to STL format.

        :param mesh: The mesh to export.
        :param output_path: The output file path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.binary:
            self._export_binary(mesh, path)
        else:
            self._export_ascii(mesh, path)

    def _export_binary(self, mesh: HPSMesh, path: Path) -> None:
        """Export mesh to binary STL format.

        :param mesh: The mesh to export.
        :param path: The output file path.
        """
        normals = self._compute_face_normals(mesh.vertices, mesh.faces)
        num_faces = mesh.num_faces

        with path.open("wb") as f:
            header = b"hpsdecode"
            f.write(header.ljust(80, b"\0"))
            f.write(struct.pack("<I", num_faces))

            for i in range(num_faces):
                normal = normals[i]
                v0, v1, v2 = mesh.vertices[mesh.faces[i]]

                f.write(struct.pack("<3f", *normal))

                f.write(struct.pack("<3f", *v0))
                f.write(struct.pack("<3f", *v1))
                f.write(struct.pack("<3f", *v2))

                f.write(b"\x00\x00")

    def _export_ascii(self, mesh: HPSMesh, path: Path) -> None:
        """Export mesh to ASCII STL format.

        :param mesh: The mesh to export.
        :param path: The output file path.
        """
        normals = self._compute_face_normals(mesh.vertices, mesh.faces)

        with path.open("w", encoding="ascii") as f:
            f.write(f"solid {path.stem}\n")

            for i in range(mesh.num_faces):
                normal = normals[i]
                v0, v1, v2 = mesh.vertices[mesh.faces[i]]

                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")

            f.write(f"endsolid {path.stem}\n")

    @staticmethod
    def _compute_face_normals(
        vertices: npt.NDArray[np.floating],
        faces: npt.NDArray[np.integer],
    ) -> npt.NDArray[np.floating]:
        """Compute unit normal vectors for each face.

        :param vertices: The vertices of the mesh of shape (N, 3).
        :param faces: The face indices of shape (M, 3).
        :return: The normal vectors for each face of shape (M, 3).
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        normals = np.cross(v1 - v0, v2 - v0)

        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-10)
        normals /= lengths

        return normals.astype(np.float32)
