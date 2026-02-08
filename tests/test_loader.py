import io
from pathlib import Path

import pytest

from hpsdecode import load_hps
from hpsdecode.exceptions import HPSParseError, HPSSchemaError

#: An HPS 'Packed_geometry' XML string representing a mesh with three vertices and one triangle.
SIMPLE_PACKED_GEOMETRY_XML = """
<Packed_geometry>
    <Schema>CA</Schema>
    <Binary_data>
        <CA version="1.0">
            <Vertices base64_encoded_bytes="36" vertex_count="3">AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA</Vertices>
            <Facets base64_encoded_bytes="1" facet_count="1">BA==</Facets>
        </CA>
    </Binary_data>
</Packed_geometry>
"""

#: An HPS XML string representing a mesh with three vertices and one triangle.
SIMPLE_MESH_XML = f"""
<HPS version="1.1">
    {SIMPLE_PACKED_GEOMETRY_XML}
    <Properties>
        <Property name="TestProp" value="TestValue"/>
    </Properties>
</HPS>
"""

#: An HPS XML string representing a colored mesh with three vertices and one triangle.
COLORED_MESH_XML = """
<HPS version="1.1">
    <Packed_geometry>
        <Schema>CC</Schema>
        <Binary_data>
            <CC version="1.0">
                <Vertices base64_encoded_bytes="36" vertex_count="3">AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA</Vertices>
                <Facets base64_encoded_bytes="1" facet_count="1" color="16744512">BA==</Facets>
            </CC>
        </Binary_data>
    </Packed_geometry>
    <VertexColorSets>
        <VertexColorSet Id="fluorescence" Base64EncodedBytes="9">/wAAAP8AAAD/</VertexColorSet>
    </VertexColorSets>
</HPS>
"""

#: An HPS XML string representing a textured mesh with UVs and a PNG texture image.
TEXTURED_MESH_XML = f"""
<HPS version="1.1">
    {SIMPLE_PACKED_GEOMETRY_XML}
    <TextureData2>
        <PerVertexTextureCoord Base64EncodedBytes="15" Key="1" TextureCoordId="0">AQAAAAAB/38AAAEAAP9/</PerVertexTextureCoord>
        <TextureImages>
            <TextureImage Version="2" Width="2" Height="2" TextureName="Texture" Base64EncodedBytes="75" RefTextureCoordId="0" Id="0" TextureCoordSet="0">iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGP83+DAwMDAxAAGABXdAcOKp/0nAAAAAElFTkSuQmCC</TextureImage>
        </TextureImages>
    </TextureData2>
</HPS>
"""

#: An HPS XML string representing a mesh with a single spline.
MESH_WITH_SPLINE_XML = f"""
<HPS version="1.1">
    {SIMPLE_PACKED_GEOMETRY_XML}
    <Splines>
        <Object name="Spline" type="TSysSpline">
            <Property name="iMisc1" value="12"/>
            <Property name="Closed" value="False"/>
            <Property name="Radius" value="0.25"/>
            <Property name="Color" value="16737380"/>
            <Property name="Name" value="Test Spline"/>
            <ControlPointsPacked>AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAPwAAgD8AAAAA</ControlPointsPacked>
        </Object>
    </Splines>
</HPS>
"""

#: An HPS XML string representing a mesh with a cyclic spline.
MESH_WITH_CYCLIC_SPLINE_XML = f"""
<HPS version="1.1">
    {SIMPLE_PACKED_GEOMETRY_XML}
    <Splines>
        <Object name="Spline" type="TSysSpline">
            <Property name="iMisc1" value="5"/>
            <Property name="Closed" value="True"/>
            <Property name="Radius" value="0.1"/>
            <Property name="Color" value="255"/>
            <Property name="Name" value="Cyclic Spline"/>
            <ControlPointsPacked>AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAPwAAgD8AAAAA</ControlPointsPacked>
        </Object>
    </Splines>
</HPS>
"""

#: An HPS XML string representing a mesh with multiple splines.
MESH_WITH_MULTIPLE_SPLINES_XML = f"""
<HPS version="1.1">
    {SIMPLE_PACKED_GEOMETRY_XML}
    <Splines>
        <Object name="Spline" type="TSysSpline">
            <Property name="iMisc1" value="10"/>
            <Property name="Closed" value="False"/>
            <Property name="Radius" value="0.15"/>
            <Property name="Color" value="16711680"/>
            <Property name="Name" value="First Spline"/>
            <ControlPointsPacked>AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAPwAAgD8AAAAA</ControlPointsPacked>
        </Object>
        <Object name="Spline" type="TSysSpline">
            <Property name="iMisc1" value="20"/>
            <Property name="Closed" value="True"/>
            <Property name="Radius" value="0.2"/>
            <Property name="Color" value="65280"/>
            <Property name="Name" value="Second Spline"/>
            <ControlPointsPacked>AAAAAAAAAAAAAAA/AAAAAAAAgD8AAAA/</ControlPointsPacked>
        </Object>
    </Splines>
</HPS>
"""


class TestLoadHPS:
    """Tests for the load_hps function."""

    def test_load_from_bytes_io(self) -> None:
        """Load mesh from BytesIO object."""
        packed, mesh = load_hps(io.BytesIO(SIMPLE_MESH_XML.encode()))

        assert mesh.num_vertices == 3
        assert mesh.num_faces == 1
        assert mesh.vertices.shape == (3, 3)
        assert mesh.faces.shape == (1, 3)

    def test_load_from_file_path(self, tmp_path: Path) -> None:
        """Load mesh from file path."""
        hps_file = tmp_path / "test.hps"
        hps_file.write_text(SIMPLE_MESH_XML)

        packed, mesh = load_hps(hps_file)

        assert mesh.num_vertices == 3
        assert mesh.num_faces == 1

    def test_ca_schema(self) -> None:
        """Load CA schema mesh."""
        packed, mesh = load_hps(io.BytesIO(SIMPLE_MESH_XML.encode()))

        assert packed.schema == "CA"
        assert not packed.is_encrypted

    def test_cc_schema(self) -> None:
        """Load CC schema mesh."""
        packed, mesh = load_hps(io.BytesIO(COLORED_MESH_XML.encode()))

        assert packed.schema == "CC"
        assert not packed.is_encrypted

    def test_load_properties(self) -> None:
        """Load properties from HPS file."""
        packed, _ = load_hps(io.BytesIO(SIMPLE_MESH_XML.encode()))

        assert "TestProp" in packed.properties
        assert packed.properties["TestProp"] == "TestValue"

    def test_unsupported_schema_raises_error(self) -> None:
        """Raise HPSSchemaError for unsupported schema."""
        unsupported_xml = """
        <HPS version="1.1">
            <Packed_geometry>
                <Schema>ZZ</Schema>
                <Binary_data>
                    <ZZ version="1.0">
                        <Vertices base64_encoded_bytes="0" vertex_count="0"></Vertices>
                        <Facets base64_encoded_bytes="0" facet_count="0"></Facets>
                    </ZZ>
                </Binary_data>
            </Packed_geometry>
        </HPS>
        """

        with pytest.raises(HPSSchemaError):
            load_hps(io.BytesIO(unsupported_xml.encode()))

    def test_missing_schema_raises_error(self) -> None:
        """Raise HPSParseError when schema element is missing."""
        missing_schema_xml = """
        <HPS version="1.1">
            <Packed_geometry>
                <Binary_data></Binary_data>
            </Packed_geometry>
        </HPS>
        """

        with pytest.raises(HPSParseError, match="Required XML element"):
            load_hps(io.BytesIO(missing_schema_xml.encode()))

    def test_missing_vertices_raises_error(self) -> None:
        """Raise HPSParseError when vertices element is missing."""
        missing_vertices_xml = """
        <HPS version="1.1">
            <Packed_geometry>
                <Schema>CC</Schema>
                <Binary_data>
                    <CC version="1.0">
                        <Facets base64_encoded_bytes="1" facet_count="1">BA==</Facets>
                    </CC>
                </Binary_data>
            </Packed_geometry>
        </HPS>
        """

        with pytest.raises(HPSParseError, match="Required XML element"):
            load_hps(io.BytesIO(missing_vertices_xml.encode()))

    def test_missing_facets_raises_error(self) -> None:
        """Raise HPSParseError when facets element is missing."""
        missing_facets_xml = """
        <HPS version="1.1">
            <Packed_geometry>
                <Schema>CC</Schema>
                <Binary_data>
                    <CC version="1.0">
                        <Vertices base64_encoded_bytes="36" vertex_count="3">AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA</Vertices>
                    </CC>
                </Binary_data>
            </Packed_geometry>
        </HPS>
        """

        with pytest.raises(HPSParseError, match="Required XML element"):
            load_hps(io.BytesIO(missing_facets_xml.encode()))

    def test_vertex_count_mismatch_raises_error(self) -> None:
        """Raise HPSParseError when vertex count doesn't match data."""
        mismatch_xml = """
        <HPS version="1.1">
            <Packed_geometry>
                <Schema>CC</Schema>
                <Binary_data>
                    <CC version="1.0">
                        <Vertices base64_encoded_bytes="36" vertex_count="5">AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA</Vertices>
                        <Facets base64_encoded_bytes="1" facet_count="1">BA==</Facets>
                    </CC>
                </Binary_data>
            </Packed_geometry>
        </HPS>
        """

        with pytest.raises(HPSParseError, match="Vertex count mismatch"):
            load_hps(io.BytesIO(mismatch_xml.encode()))

    def test_face_count_mismatch_raises_error(self) -> None:
        """Raise HPSParseError when face count doesn't match data."""
        mismatch_xml = """
        <HPS version="1.1">
            <Packed_geometry>
                <Schema>CC</Schema>
                <Binary_data>
                    <CC version="1.0">
                        <Vertices base64_encoded_bytes="36" vertex_count="3">AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA</Vertices>
                        <Facets base64_encoded_bytes="1" facet_count="5">BA==</Facets>
                    </CC>
                </Binary_data>
            </Packed_geometry>
        </HPS>
        """

        with pytest.raises(HPSParseError, match="Face count mismatch"):
            load_hps(io.BytesIO(mismatch_xml.encode()))


class TestLoadHPSWithColors:
    """Tests for loading meshes with vertex and face colors."""

    def test_load_colored_mesh(self) -> None:
        """Load mesh with vertex and face colors."""
        packed, mesh = load_hps(io.BytesIO(COLORED_MESH_XML.encode()))

        assert mesh.num_vertices == 3
        assert mesh.has_vertex_colors
        assert mesh.has_face_colors
        assert mesh.vertex_colors.shape == (3, 3)
        assert mesh.face_colors.shape == (1, 3)

    def test_parse_face_colors_from_attribute(self) -> None:
        """Parse face colors from facet ``color`` attribute correctly."""
        packed, mesh = load_hps(io.BytesIO(COLORED_MESH_XML.encode()))

        # Color 16744512 = 0xFF8040 = RGB(255, 128, 64)
        assert mesh.face_colors[0, 0] == 255
        assert mesh.face_colors[0, 1] == 128
        assert mesh.face_colors[0, 2] == 64


class TestLoadHPSWithTextures:
    """Tests for loading meshes with texture coordinates and images."""

    def test_load_textured_mesh(self) -> None:
        """Load mesh with texture coordinates and images."""
        packed, mesh = load_hps(io.BytesIO(TEXTURED_MESH_XML.encode()))

        assert mesh.num_vertices == 3
        assert mesh.has_texture_coords
        assert mesh.has_textures
        assert len(mesh.texture_images) == 1

    def test_parse_texture_coordinates(self) -> None:
        """Parse texture coordinates from binary data correctly."""
        packed, mesh = load_hps(io.BytesIO(TEXTURED_MESH_XML.encode()))

        assert mesh.uv.shape[0] > 0
        assert mesh.uv.shape[1] == 2


class TestLoadHPSWithSplines:
    """Tests for loading meshes with spline data."""

    def test_load_mesh_with_spline(self) -> None:
        """Load mesh with a single spline."""
        packed, mesh = load_hps(io.BytesIO(MESH_WITH_SPLINE_XML.encode()))

        assert mesh.has_splines
        assert len(mesh.splines) == 1

    def test_parse_spline_name(self) -> None:
        """Parse spline name property correctly."""
        _, mesh = load_hps(io.BytesIO(MESH_WITH_SPLINE_XML.encode()))

        spline = mesh.splines[0]
        assert spline.name == "Test Spline"

    def test_parse_spline_radius(self) -> None:
        """Parse spline radius property correctly."""
        _, mesh = load_hps(io.BytesIO(MESH_WITH_SPLINE_XML.encode()))

        spline = mesh.splines[0]
        assert spline.radius == pytest.approx(0.25)

    def test_parse_spline_color(self) -> None:
        """Parse spline color property correctly."""
        _, mesh = load_hps(io.BytesIO(MESH_WITH_SPLINE_XML.encode()))

        spline = mesh.splines[0]
        assert spline.color == 16737380

    def test_parse_spline_misc(self) -> None:
        """Parse spline misc property correctly."""
        _, mesh = load_hps(io.BytesIO(MESH_WITH_SPLINE_XML.encode()))

        spline = mesh.splines[0]
        assert spline.misc == 12

    def test_parse_spline_open(self) -> None:
        """Parse open spline correctly."""
        _, mesh = load_hps(io.BytesIO(MESH_WITH_SPLINE_XML.encode()))

        spline = mesh.splines[0]
        assert not spline.is_cyclic

    def test_parse_spline_closed(self) -> None:
        """Parse cyclic spline correctly."""
        _, mesh = load_hps(io.BytesIO(MESH_WITH_CYCLIC_SPLINE_XML.encode()))

        spline = mesh.splines[0]
        assert spline.is_cyclic

    def test_parse_spline_control_points(self) -> None:
        """Parse spline control points correctly."""
        _, mesh = load_hps(io.BytesIO(MESH_WITH_SPLINE_XML.encode()))

        spline = mesh.splines[0]
        assert spline.num_control_points == 3
        assert spline.control_points.shape == (3, 3)
        assert spline.control_points[0, 0] == pytest.approx(0.0)
        assert spline.control_points[1, 0] == pytest.approx(1.0)
        assert spline.control_points[2, 0] == pytest.approx(0.5)

    def test_load_multiple_splines(self) -> None:
        """Load mesh with multiple splines."""
        _, mesh = load_hps(io.BytesIO(MESH_WITH_MULTIPLE_SPLINES_XML.encode()))

        assert mesh.has_splines
        assert len(mesh.splines) == 2

    def test_parse_multiple_splines_properties(self) -> None:
        """Parse properties from multiple splines correctly."""
        _, mesh = load_hps(io.BytesIO(MESH_WITH_MULTIPLE_SPLINES_XML.encode()))

        first_spline = mesh.splines[0]
        second_spline = mesh.splines[1]

        assert first_spline.name == "First Spline"
        assert first_spline.radius == pytest.approx(0.15)
        assert first_spline.color == 16711680
        assert first_spline.misc == 10
        assert not first_spline.is_cyclic

        assert second_spline.name == "Second Spline"
        assert second_spline.radius == pytest.approx(0.2)
        assert second_spline.color == 65280
        assert second_spline.misc == 20
        assert second_spline.is_cyclic

    def test_mesh_without_splines(self) -> None:
        """Load mesh without splines correctly."""
        _, mesh = load_hps(io.BytesIO(SIMPLE_MESH_XML.encode()))

        assert not mesh.has_splines
        assert len(mesh.splines) == 0
