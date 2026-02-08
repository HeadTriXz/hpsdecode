"""OBJ format exporter."""

from __future__ import annotations

__all__ = ["MaterialConfig", "OBJExporter"]

import dataclasses
import io
import typing as t
from pathlib import Path

import numpy as np
from PIL import Image

from hpsdecode.export.base import BaseExporter
from hpsdecode.texture import deduplicate_vertices_for_uv, face_colors_to_vertex_colors

if t.TYPE_CHECKING:
    import os

    import numpy.typing as npt

    from hpsdecode.mesh import HPSMesh


@dataclasses.dataclass
class MaterialConfig:
    """Material properties for OBJ MTL files."""

    #: Ambient color (Ka).
    ambient: tuple[float, float, float] = (0.2, 0.2, 0.2)

    #: Diffuse color (Kd).
    diffuse: tuple[float, float, float] = (0.8, 0.8, 0.8)

    #: Specular color (Ks).
    specular: tuple[float, float, float] = (1.0, 1.0, 1.0)

    #: Specular exponent (Ns).
    shininess: float = 32.0

    #: Optical density (Ni). Default is 1.0 (no refraction).
    optical_density: float = 1.0

    #: Dissolve (d). 1.0 = fully opaque, 0.0 = fully transparent.
    dissolve: float = 1.0


class OBJExporter(BaseExporter):
    """Export meshes to OBJ format with optional materials and textures."""

    #: Material configuration for MTL file.
    material: MaterialConfig

    #: Whether to include vertex colors if available.
    include_colors: bool

    #: Whether to include textures if available.
    include_textures: bool

    def __init__(
        self,
        material: MaterialConfig | None = None,
        include_colors: bool = True,
        include_textures: bool = True,
    ) -> None:
        """Initialize the OBJ exporter.

        :param material: Material configuration for MTL file. If ``None``, default values are used.
        :param include_colors: Whether to include vertex colors if available. Default is ``True``.
        :param include_textures: Whether to include textures if available. Default is ``True``.
        """
        self.material = material or MaterialConfig()
        self.include_colors = include_colors
        self.include_textures = include_textures

    def export(self, mesh: HPSMesh, output_path: str | os.PathLike[str]) -> None:
        """Export a mesh to OBJ format.

        :param mesh: The mesh to export.
        :param output_path: The output file path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        use_texture = mesh.has_textures and mesh.has_texture_coords and self.include_textures
        use_vertex_colors = mesh.has_vertex_colors and self.include_colors
        use_face_colors = mesh.has_face_colors and self.include_colors

        if use_texture:
            self._export_with_textures(mesh, path)
        else:
            vertex_colors = None
            if use_vertex_colors:
                vertex_colors = mesh.vertex_colors
            elif use_face_colors:
                vertex_colors = face_colors_to_vertex_colors(mesh)

            self._export_geometry(mesh, path, vertex_colors)

    def _export_geometry(
        self,
        mesh: HPSMesh,
        path: Path,
        vertex_colors: npt.NDArray[np.uint8] | None,
    ) -> None:
        """Export mesh geometry with optional vertex colors.

        :param mesh: The mesh to export.
        :param path: The output file path.
        :param vertex_colors: Optional vertex colors (N, 3).
        """
        has_colors = vertex_colors is not None and vertex_colors.size > 0

        with path.open("w", encoding="utf-8") as f:
            f.write("# hpsdecode\n")

            for i, v in enumerate(mesh.vertices):
                if has_colors:
                    color = vertex_colors[i] / 255.0
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {color[0]:.4f} {color[1]:.4f} {color[2]:.4f}\n")
                else:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write("\n")

            for face in mesh.faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    def _export_with_textures(self, mesh: HPSMesh, path: Path) -> None:
        """Export mesh with texture mapping.

        :param mesh: The mesh to export.
        :param path: The output file path.
        """
        mtl_name = path.stem
        mtl_path = path.with_suffix(".mtl")
        texture_name = f"{path.stem}_texture.png"
        texture_path = path.parent / texture_name

        num_faces = mesh.num_faces
        uv_coords = mesh.uv.reshape(num_faces, 3, 2)

        new_vertices, new_uvs, new_faces = deduplicate_vertices_for_uv(mesh.vertices, mesh.faces, uv_coords)

        with path.open("w", encoding="utf-8") as f:
            f.write("# hpsdecode\n")
            f.write(f"mtllib {mtl_path.name}\n\n")

            for v in new_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write("\n")

            for uv in new_uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            f.write("\n")

            f.write(f"usemtl {mtl_name}\n")
            for face in new_faces:
                f.write(f"f {face[0] + 1}/{face[0] + 1} {face[1] + 1}/{face[1] + 1} {face[2] + 1}/{face[2] + 1}\n")

        self._write_mtl_file(mtl_path, mtl_name, texture_name)
        self._write_texture_image(mesh.texture_images[0], texture_path)

    def _write_mtl_file(self, mtl_path: Path, material_name: str, texture_filename: str) -> None:
        """Write MTL material definition file.

        :param mtl_path: The output path for the MTL file.
        :param material_name: The name of the material to define.
        :param texture_filename: The filename of the texture image to reference.
        """
        mat = self.material

        with mtl_path.open("w", encoding="utf-8") as f:
            f.write("# hpsdecode\n\n")
            f.write(f"newmtl {material_name}\n")
            f.write(f"Ka {mat.ambient[0]:.4f} {mat.ambient[1]:.4f} {mat.ambient[2]:.4f}\n")
            f.write(f"Kd {mat.diffuse[0]:.4f} {mat.diffuse[1]:.4f} {mat.diffuse[2]:.4f}\n")
            f.write(f"Ks {mat.specular[0]:.4f} {mat.specular[1]:.4f} {mat.specular[2]:.4f}\n")
            f.write(f"Ns {mat.shininess:.4f}\n")
            f.write(f"Ni {mat.optical_density:.4f}\n")
            f.write(f"d {mat.dissolve:.4f}\n")
            f.write(f"map_Kd {texture_filename}\n")

    @staticmethod
    def _write_texture_image(texture_data: bytes, texture_path: Path) -> None:
        """Write the texture image.

        :param texture_data: The raw texture image data (BGR format).
        :param texture_path: The output path for the texture image (PNG format).
        """
        img = Image.open(io.BytesIO(texture_data))
        if img.mode != "RGB":
            img = img.convert("RGB")

        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))

        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        img.save(texture_path, "PNG")
