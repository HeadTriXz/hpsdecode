from hpsdecode.mesh import HPSMesh


class TestHPSMesh:
    """Tests for the HPSMesh class."""

    def test_num_vertices(self, simple_mesh: HPSMesh) -> None:
        """Return vertex count."""
        assert simple_mesh.num_vertices == 3
        assert len(simple_mesh.vertices) == 3

    def test_num_faces(self, simple_mesh: HPSMesh) -> None:
        """Return face count."""
        assert simple_mesh.num_faces == 1
        assert len(simple_mesh.faces) == 1

    def test_has_texture_coords_with_uv(self, textured_mesh: HPSMesh) -> None:
        """Return ``True`` when UV data present."""
        assert textured_mesh.has_texture_coords

    def test_has_texture_coords_without_uv(self, simple_mesh: HPSMesh) -> None:
        """Return ``False`` when UV array empty."""
        assert not simple_mesh.has_texture_coords

    def test_has_vertex_colors_with_colors(self, colored_mesh: HPSMesh) -> None:
        """Return ``True`` when vertex colors present."""
        assert colored_mesh.has_vertex_colors

    def test_has_vertex_colors_without_colors(self, simple_mesh: HPSMesh) -> None:
        """Return ``False`` when vertex colors empty."""
        assert not simple_mesh.has_vertex_colors

    def test_has_face_colors_with_colors(self, colored_mesh: HPSMesh) -> None:
        """Return ``True`` when face colors present."""
        assert colored_mesh.has_face_colors

    def test_has_face_colors_without_colors(self, simple_mesh: HPSMesh) -> None:
        """Return ``False`` when face colors empty."""
        assert not simple_mesh.has_face_colors

    def test_has_textures_with_images(self, textured_mesh: HPSMesh) -> None:
        """Return ``True`` when texture images present."""
        assert textured_mesh.has_textures

    def test_has_textures_without_images(self, simple_mesh: HPSMesh) -> None:
        """Return ``False`` when texture images list empty."""
        assert not simple_mesh.has_textures

    def test_empty_mesh(self, empty_mesh: HPSMesh) -> None:
        """Handle empty mesh with no geometry or attributes."""
        assert empty_mesh.num_vertices == 0
        assert empty_mesh.num_faces == 0
        assert not empty_mesh.has_texture_coords
        assert not empty_mesh.has_vertex_colors
        assert not empty_mesh.has_face_colors
        assert not empty_mesh.has_textures
