import struct
from pathlib import Path

import pytest

from hpsdecode.export import ExportFormat, export_mesh
from hpsdecode.export.obj import MaterialConfig, OBJExporter
from hpsdecode.export.ply import PLYExporter
from hpsdecode.export.stl import STLExporter
from hpsdecode.mesh import HPSMesh


class TestSTLExporter:
    """Tests for the STL format exporter."""

    def test_binary_header_structure(self, simple_mesh: HPSMesh, tmp_path: Path) -> None:
        """Write 80-byte header and triangle count in binary format."""
        exporter = STLExporter(binary=True)
        output_path = tmp_path / "test.stl"

        exporter.export(simple_mesh, output_path)

        with output_path.open("rb") as f:
            header = f.read(80)
            num_triangles = struct.unpack("<I", f.read(4))[0]

        assert len(header) == 80
        assert num_triangles == 1

    def test_binary_triangle_structure(self, simple_mesh: HPSMesh, tmp_path: Path) -> None:
        """Write a triangle with normal and vertex coordinates in binary format."""
        exporter = STLExporter(binary=True)
        output_path = tmp_path / "test.stl"

        exporter.export(simple_mesh, output_path)

        with output_path.open("rb") as f:
            f.seek(84)
            normal = struct.unpack("<3f", f.read(12))
            v0 = struct.unpack("<3f", f.read(12))
            v1 = struct.unpack("<3f", f.read(12))
            v2 = struct.unpack("<3f", f.read(12))
            attr = f.read(2)

        assert len(normal) == 3
        assert len(v0) == 3
        assert len(v1) == 3
        assert len(v2) == 3
        assert attr == b"\x00\x00"

    def test_compute_normal_vector(self, simple_mesh: HPSMesh, tmp_path: Path) -> None:
        """Compute correct normal for XY plane triangle."""
        exporter = STLExporter(binary=True)
        output_path = tmp_path / "test.stl"

        exporter.export(simple_mesh, output_path)

        with output_path.open("rb") as f:
            f.seek(84)
            normal = struct.unpack("<3f", f.read(12))

        assert normal[0] == pytest.approx(0.0)
        assert normal[1] == pytest.approx(0.0)
        assert normal[2] == pytest.approx(1.0)

    def test_ascii_format(self, simple_mesh: HPSMesh, tmp_path: Path) -> None:
        """Write valid ASCII STL with correct structure."""
        exporter = STLExporter(binary=False)
        output_path = tmp_path / "test.stl"

        exporter.export(simple_mesh, output_path)

        with output_path.open("r") as f:
            content = f.read()

        assert content.startswith("solid test")
        assert "facet normal" in content
        assert "outer loop" in content
        assert "vertex" in content
        assert "endloop" in content
        assert "endfacet" in content
        assert content.endswith("endsolid test\n")


class TestPLYExporter:
    """Tests for the PLY format exporter."""

    def test_binary_header_no_colors(self, simple_mesh: HPSMesh, tmp_path: Path) -> None:
        """Write binary header with geometry only."""
        exporter = PLYExporter(binary=True, include_colors=False)
        output_path = tmp_path / "test.ply"

        exporter.export(simple_mesh, output_path)

        with output_path.open("rb") as f:
            content = f.read().decode("ascii", errors="ignore")

        assert "ply" in content
        assert "format binary_little_endian 1.0" in content
        assert "element vertex 3" in content
        assert "element face 1" in content
        assert "property float x" in content

        assert "property uchar red" not in content
        assert "property uchar green" not in content
        assert "property uchar blue" not in content

    def test_binary_header_with_colors(self, colored_mesh: HPSMesh, tmp_path: Path) -> None:
        """Write binary header with RGB color properties."""
        exporter = PLYExporter(binary=True, include_colors=True)
        output_path = tmp_path / "test.ply"

        exporter.export(colored_mesh, output_path)

        with output_path.open("rb") as f:
            header = f.read(500).decode("ascii", errors="ignore")

        assert "property uchar red" in header
        assert "property uchar green" in header
        assert "property uchar blue" in header

    def test_ascii_format(self, simple_mesh: HPSMesh, tmp_path: Path) -> None:
        """Write ASCII format with header and numeric data."""
        exporter = PLYExporter(binary=False, include_colors=False)
        output_path = tmp_path / "test.ply"

        exporter.export(simple_mesh, output_path)

        with output_path.open("r") as f:
            content = f.read()

        lines = content.split("\n")
        vertex_lines = [line for line in lines if line and line[0].isdigit() and "." in line]

        assert "ply" in content
        assert "format ascii 1.0" in content
        assert "end_header" in content
        assert len(vertex_lines) >= 3

    def test_ascii_with_colors(self, colored_mesh: HPSMesh, tmp_path: Path) -> None:
        """Write ASCII vertex lines with RGB values."""
        exporter = PLYExporter(binary=False, include_colors=True)
        output_path = tmp_path / "test.ply"

        exporter.export(colored_mesh, output_path)

        with output_path.open("r") as f:
            lines = f.readlines()

        vertex_line = None
        for line in lines:
            if line and line[0].isdigit() and "." in line:
                vertex_line = line.strip()
                break

        parts = vertex_line.split()

        assert vertex_line is not None
        assert len(parts) == 6

    def test_bake_texture_to_colors(self, textured_mesh: HPSMesh, tmp_path: Path) -> None:
        """Bake texture to vertex colors in PLY."""
        exporter = PLYExporter(binary=False, include_textures=True)
        output_path = tmp_path / "test.ply"

        exporter.export(textured_mesh, output_path)

        with output_path.open("r") as f:
            content = f.read()

        assert "property uchar red" in content
        assert "property uchar green" in content
        assert "property uchar blue" in content


class TestOBJExporter:
    """Tests for the OBJ format exporter."""

    def test_basic_geometry(self, simple_mesh: HPSMesh, tmp_path: Path) -> None:
        """Write vertices and faces in OBJ format."""
        exporter = OBJExporter(include_colors=False, include_textures=False)
        output_path = tmp_path / "test.obj"

        exporter.export(simple_mesh, output_path)

        with output_path.open("r") as f:
            content = f.read()

        assert content.count("v ") == 3
        assert content.count("f ") == 1

    def test_vertex_colors(self, colored_mesh: HPSMesh, tmp_path: Path) -> None:
        """Write vertex lines with RGB color values."""
        exporter = OBJExporter(include_colors=True, include_textures=False)
        output_path = tmp_path / "test.obj"

        exporter.export(colored_mesh, output_path)

        with output_path.open("r") as f:
            content = f.read()

        vertex_lines = [line for line in content.split("\n") if line.startswith("v ")]
        parts = vertex_lines[0].split()

        assert len(vertex_lines) == 3
        assert len(parts) == 7  # v x y z r g b

    def test_texture_files_created(self, textured_mesh: HPSMesh, tmp_path: Path) -> None:
        """Create OBJ, MTL, and PNG files for textured mesh."""
        exporter = OBJExporter(include_textures=True)
        output_path = tmp_path / "test.obj"

        exporter.export(textured_mesh, output_path)

        assert output_path.exists()
        assert (tmp_path / "test.mtl").exists()
        assert (tmp_path / "test.png").exists()

    def test_texture_references(self, textured_mesh: HPSMesh, tmp_path: Path) -> None:
        """Reference MTL file and include UV coordinates."""
        exporter = OBJExporter(include_textures=True)
        output_path = tmp_path / "test.obj"

        exporter.export(textured_mesh, output_path)

        with output_path.open("r") as f:
            content = f.read()

        assert "mtllib test.mtl" in content
        assert "vt " in content
        assert "usemtl test" in content
        assert "f " in content and "/" in content

    def test_mtl_material_properties(self, textured_mesh: HPSMesh, tmp_path: Path) -> None:
        """Write MTL with material properties and texture map."""
        exporter = OBJExporter(include_textures=True)
        output_path = tmp_path / "test.obj"

        exporter.export(textured_mesh, output_path)

        mtl_path = tmp_path / "test.mtl"
        with mtl_path.open("r") as f:
            content = f.read()

        assert "newmtl test" in content
        assert "Ka " in content
        assert "Kd " in content
        assert "Ks " in content
        assert "Ns " in content
        assert "map_Kd test.png" in content

    def test_custom_material(self, textured_mesh: HPSMesh, tmp_path: Path) -> None:
        """Apply custom MaterialConfig to MTL file."""
        material = MaterialConfig(
            ambient=(0.1, 0.1, 0.1),
            diffuse=(0.9, 0.9, 0.9),
            specular=(0.5, 0.5, 0.5),
            shininess=64.0,
        )
        exporter = OBJExporter(material=material, include_textures=True)
        output_path = tmp_path / "test.obj"

        exporter.export(textured_mesh, output_path)

        mtl_path = tmp_path / "test.mtl"
        with mtl_path.open("r") as f:
            content = f.read()

        assert "Ns 64.0" in content


class TestExportFormat:
    """Tests for ExportFormat enum and extension parsing."""

    def test_from_extension_stl(self) -> None:
        """Recognize the ``.stl`` extension."""
        fmt = ExportFormat.from_extension("model.stl")

        assert fmt == ExportFormat.STL

    def test_from_extension_obj(self) -> None:
        """Recognize the ``.obj`` extension."""
        fmt = ExportFormat.from_extension("model.obj")

        assert fmt == ExportFormat.OBJ

    def test_from_extension_ply(self) -> None:
        """Recognize the ``.ply`` extension."""
        fmt = ExportFormat.from_extension("model.ply")

        assert fmt == ExportFormat.PLY

    def test_from_extension_case_insensitive(self) -> None:
        """Handle uppercase extensions."""
        fmt = ExportFormat.from_extension("model.STL")

        assert fmt == ExportFormat.STL

    def test_from_extension_unsupported(self) -> None:
        """Raise ValueError for unsupported extensions."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            ExportFormat.from_extension("model.fbx")

    def test_from_extension_path_object(self) -> None:
        """Accept :py:class:`pathlib.Path` objects."""
        fmt = ExportFormat.from_extension(Path("/tmp/model.ply"))

        assert fmt == ExportFormat.PLY


class TestExportMesh:
    """Tests for the export_mesh function."""

    def test_auto_detect_format(self, simple_mesh: HPSMesh, tmp_path: Path) -> None:
        """Detect format from file extension."""
        output_path = tmp_path / "test.stl"

        export_mesh(simple_mesh, output_path)

        assert output_path.exists()

    def test_explicit_format(self, simple_mesh: HPSMesh, tmp_path: Path) -> None:
        """Use explicit format parameter over extension."""
        output_path = tmp_path / "test.txt"

        export_mesh(simple_mesh, output_path, export_format=ExportFormat.PLY)

        assert output_path.exists()

    def test_forward_options(self, colored_mesh: HPSMesh, tmp_path: Path) -> None:
        """Forward options to exporter."""
        output_path = tmp_path / "test.ply"

        export_mesh(colored_mesh, output_path, export_format=ExportFormat.PLY, binary=False, include_colors=True)

        with output_path.open("r") as f:
            content = f.read()

        assert "format ascii" in content
        assert "property uchar red" in content
        assert "property uchar green" in content
        assert "property uchar blue" in content


class TestMeshExportMethod:
    """Tests for the HPSMesh.export method."""

    def test_export_method(self, simple_mesh: HPSMesh, tmp_path: Path) -> None:
        """Export via mesh instance method."""
        output_path = tmp_path / "test.ply"

        simple_mesh.export(output_path)

        assert output_path.exists()

    def test_export_method_with_options(self, colored_mesh: HPSMesh, tmp_path: Path) -> None:
        """Forward options through mesh export method."""
        output_path = tmp_path / "test.ply"

        colored_mesh.export(output_path, binary=False, include_colors=True)

        with output_path.open("r") as f:
            content = f.read()

        assert "format ascii" in content
        assert "property uchar red" in content
        assert "property uchar green" in content
        assert "property uchar blue" in content
