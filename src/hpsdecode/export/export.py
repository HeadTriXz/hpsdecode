"""Export HPS mesh data to common 3D file formats."""

from __future__ import annotations

__all__ = ["ExportFormat", "export_mesh"]

import enum
import typing as t
from pathlib import Path

from hpsdecode.export.obj import OBJExporter
from hpsdecode.export.ply import PLYExporter
from hpsdecode.export.stl import STLExporter

if t.TYPE_CHECKING:
    import os

    from hpsdecode.mesh import HPSMesh


class ExportFormat(enum.Enum):
    """Supported export formats."""

    STL = "stl"
    OBJ = "obj"
    PLY = "ply"

    @classmethod
    def from_extension(cls, path: str | os.PathLike[str]) -> ExportFormat:
        """Determine export format from file extension.

        :param path: The file path.
        :return: The corresponding export format.
        :raises ValueError: If the extension is not supported.
        """
        ext = Path(path).suffix.lower().lstrip(".")
        try:
            return cls(ext)
        except ValueError:
            supported = ", ".join(f.value for f in cls)
            raise ValueError(f"Unsupported file extension '.{ext}'. Supported: {supported}") from None


def export_mesh(
    mesh: HPSMesh,
    output_path: str | os.PathLike[str],
    export_format: ExportFormat | None = None,
    *,
    binary: bool = True,
    include_colors: bool = True,
    include_textures: bool = True,
) -> None:
    """Export an HPS mesh to a file.

    :param mesh: The mesh to export.
    :param output_path: The output file path.
    :param export_format: The export format. If None, inferred from file extension.
    :param binary: Whether to use binary format (STL/PLY only). Default is ``True``.
    :param include_colors: Whether to include vertex colors if available (OBJ/PLY only). Default is ``True``.
    :param include_textures: Whether to include textures if available (OBJ only)
        or bake textures to vertex colors (PLY only). Default is ``True``.
    :raises ValueError: If the format is unsupported or incompatible with options.
    """
    if export_format is None:
        export_format = ExportFormat.from_extension(output_path)

    if export_format == ExportFormat.STL:
        exporter = STLExporter(binary=binary)
    elif export_format == ExportFormat.OBJ:
        exporter = OBJExporter(include_colors=include_colors, include_textures=include_textures)
    elif export_format == ExportFormat.PLY:
        exporter = PLYExporter(binary=binary, include_textures=include_textures)
    else:
        raise ValueError(f"Unsupported export format: {export_format}")

    exporter.export(mesh, output_path)

