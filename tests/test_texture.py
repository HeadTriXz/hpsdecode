import numpy as np
import pytest

from hpsdecode.mesh import HPSMesh
from hpsdecode.texture import (
    decompress_texture_coord,
    deduplicate_vertices_for_uv,
    face_colors_to_vertex_colors,
    parse_texture_coords,
    texture_to_vertex_colors,
)


class TestTextureCoordDecompression:
    """Tests for decompressing 32-bit packed texture coordinates."""

    def test_decompress_inside_range_zero(self) -> None:
        """Decompress 0x00000000 to (0.0, 0.0)."""
        compressed = 0x00000000

        u, v = decompress_texture_coord(compressed)

        assert u == 0.0
        assert v == 0.0

    def test_decompress_inside_range_max(self) -> None:
        """Decompress 0x7FFF7FFF to approximately (1.0, 1.0)."""
        compressed = 0x7FFF7FFF

        u, v = decompress_texture_coord(compressed)

        assert u == pytest.approx(1.0, abs=1e-4)
        assert v == pytest.approx(1.0, abs=1e-4)

    def test_decompress_inside_range_half(self) -> None:
        """Decompress 0x3FFF3FFF to approximately (0.5, 0.5)."""
        compressed = 0x3FFF3FFF

        u, v = decompress_texture_coord(compressed)

        assert u == pytest.approx(0.5, abs=1e-4)
        assert v == pytest.approx(0.5, abs=1e-4)

    def test_decompress_outside_range_negative(self) -> None:
        """Decompress 0x80008000 to (-256.0, -256.0)."""
        compressed = 0x80008000

        u, v = decompress_texture_coord(compressed)

        assert u == -256.0
        assert v == -256.0

    def test_decompress_outside_range_positive(self) -> None:
        """Decompress 0xFFFFFFFF to approximately (256.0, 256.0)."""
        compressed = 0xFFFFFFFF

        u, v = decompress_texture_coord(compressed)

        assert u == 256.0
        assert v == 256.0

    def test_decompress_mixed_ranges(self) -> None:
        """Decompress mixed inside/outside ranges."""
        compressed = 0x80003FFF

        u, v = decompress_texture_coord(compressed)

        assert u == pytest.approx(0.5, abs=1e-4)
        assert v == -256.0


class TestParseTextureCoords:
    """Tests for parsing texture coordinates from binary data."""

    def test_parse_single_uv_per_vertex(self) -> None:
        """Handles single UV shared by all faces connected to a vertex."""
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        num_vertices = 3

        data = bytearray()
        for _ in range(num_vertices):
            data.append(1)
            data.extend((0x3FFF).to_bytes(2, "little"))
            data.extend((0x4000).to_bytes(2, "little"))

        uvs = parse_texture_coords(bytes(data), num_vertices, faces)

        assert uvs.shape == (3, 2)
        assert uvs[0, 0] == pytest.approx(0.5, abs=1e-4)
        assert uvs[1, 0] == pytest.approx(0.5, abs=1e-4)
        assert np.allclose(uvs[0], uvs[1])
        assert np.allclose(uvs[0], uvs[2])

    def test_parse_multiple_uvs_per_vertex(self) -> None:
        """Handles multiple UVs for vertices shared by multiple faces."""
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        num_vertices = 4

        data = bytearray()

        data.append(2)
        data.extend((0x0000).to_bytes(2, "little"))
        data.extend((0x0000).to_bytes(2, "little"))
        data.extend((0x1000).to_bytes(2, "little"))
        data.extend((0x1000).to_bytes(2, "little"))

        data.append(1)
        data.extend((0x2000).to_bytes(2, "little"))
        data.extend((0x2000).to_bytes(2, "little"))

        data.append(2)
        data.extend((0x3000).to_bytes(2, "little"))
        data.extend((0x3000).to_bytes(2, "little"))
        data.extend((0x4000).to_bytes(2, "little"))
        data.extend((0x4000).to_bytes(2, "little"))

        data.append(1)
        data.extend((0x5000).to_bytes(2, "little"))
        data.extend((0x5000).to_bytes(2, "little"))

        uvs = parse_texture_coords(bytes(data), num_vertices, faces)

        assert uvs.shape == (6, 2)
        assert uvs[0, 0] != uvs[3, 0]

    def test_parse_no_uv_marker(self) -> None:
        """Handles the special no-UV marker."""
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        num_vertices = 3

        data = bytearray()
        for _ in range(num_vertices):
            data.append(1)
            data.extend((0xFFFFFFFF).to_bytes(4, "little"))

        uvs = parse_texture_coords(bytes(data), num_vertices, faces)

        assert uvs.shape == (3, 2)
        assert np.all(uvs == 0.0)

    def test_parse_insufficient_data(self) -> None:
        """Raises ValueError when data ends before expected."""
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        num_vertices = 3

        data = b"\x01\x00\x00\x00\x00"

        with pytest.raises(ValueError, match="Unexpected end of texture data"):
            parse_texture_coords(data, num_vertices, faces)

    def test_parse_flag_mismatch(self) -> None:
        """Raises ValueError when flag doesn't match face count."""
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        num_vertices = 4

        data = bytearray()
        data.append(5)
        data.extend((0x1000).to_bytes(2, "little"))
        data.extend((0x1000).to_bytes(2, "little"))

        with pytest.raises(ValueError, match="Mismatch at vertex 0"):
            parse_texture_coords(bytes(data), num_vertices, faces)


class TestFaceColorsToVertexColors:
    """Tests for converting face colors to vertex colors."""

    def test_single_face_per_vertex(self, colored_mesh: HPSMesh) -> None:
        """Each vertex touching one face gets that face's color."""
        result = face_colors_to_vertex_colors(colored_mesh)

        assert result.shape == (3, 3)
        assert np.all(result == [255, 128, 64])

    def test_multiple_faces_averaging(self) -> None:
        """Vertices shared by multiple faces get averaged colors."""
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        face_colors = np.array([[100, 0, 0], [200, 0, 0]], dtype=np.uint8)
        vertex_colors = np.array([], dtype=np.uint8)

        mesh = HPSMesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
            uv=np.array([]),
        )

        result = face_colors_to_vertex_colors(mesh)

        assert result.shape == (4, 3)
        assert result[0, 0] == 100
        assert result[1, 0] == 150
        assert result[2, 0] == 150
        assert result[3, 0] == 200


class TestTextureToVertexColors:
    """Tests for sampling textures to vertex colors."""

    def test_texture_sampling(self, textured_mesh: HPSMesh) -> None:
        """Sample texture correctly at UV coordinates."""
        result = texture_to_vertex_colors(textured_mesh)

        assert result.shape == (3, 3)
        assert result[0, 0] == 64
        assert result[0, 1] == 128
        assert result[0, 2] == 255

    def test_no_texture_raises_error(self, empty_mesh: HPSMesh) -> None:
        """Raise ValueError when mesh has no textures."""
        with pytest.raises(ValueError, match="no texture images"):
            texture_to_vertex_colors(empty_mesh)


class TestDeduplicateVerticesForUV:
    """Tests for creating unique vertex-UV combinations."""

    def test_no_duplication_needed(self) -> None:
        """No duplication when each vertex has unique UV."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        uv_coords = np.array([[[0, 0], [1, 0], [0, 1]]], dtype=np.float32)

        new_vertices, new_uvs, new_faces = deduplicate_vertices_for_uv(vertices, faces, uv_coords)

        assert new_vertices.shape == (3, 3)
        assert new_uvs.shape == (3, 2)
        assert new_faces.shape == (1, 3)

    def test_duplication_for_different_uvs(self) -> None:
        """Duplicate vertices used with different UVs."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int32)
        uv_coords = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        )

        new_vertices, new_uvs, new_faces = deduplicate_vertices_for_uv(vertices, faces, uv_coords)

        assert new_vertices.shape[0] > 3
        assert new_uvs.shape[0] == new_vertices.shape[0]
        assert new_faces.shape == (2, 3)

    def test_shared_vertex_uv_combination(self) -> None:
        """Re-use same vertex-UV combination across faces."""
        vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.int32)
        uv_coords = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        )

        new_vertices, new_uvs, new_faces = deduplicate_vertices_for_uv(vertices, faces, uv_coords)

        assert new_faces[0, 0] == new_faces[0, 2]
        assert new_faces[1, 0] == new_faces[1, 1]
