"""Utilities for parsing and decoding HIMSA packed standard (HPS) files."""

from __future__ import annotations

__all__ = ["load_hps"]

import base64
import typing as t
import xml.etree.ElementTree as ET

import numpy as np

from hpsdecode.exceptions import HPSParseError, HPSSchemaError
from hpsdecode.mesh import HPSMesh, HPSPackedScan, SchemaType, Spline
from hpsdecode.schemas import SUPPORTED_SCHEMAS, EncryptedData, ParseContext, get_parser

if t.TYPE_CHECKING:
    import os

    import numpy.typing as npt

    from hpsdecode.encryption import EncryptionKeyProvider

#: List of XML paths to search for texture images that may be encrypted.
TEXTURE_IMAGE_PATHS_ENCRYPTED: t.Final[list[str]] = [
    ".//TextureData2/TextureImages/AdditionalTextureImage",
    ".//TextureData/TextureImages/AdditionalTextureImage",
    ".//PartialTextureData/TextureImages/TextureImage",
]

#: List of XML paths to search for texture images in unencrypted files.
TEXTURE_IMAGE_PATHS_UNENCRYPTED: t.Final[list[str]] = [
    ".//TextureImages/TextureImage",
]


def decode_binary_element(element: ET.Element) -> bytes:
    """Decode base64-encoded binary data from an XML element.

    :param element: The XML element containing base64-encoded data.
    :return: The decoded binary data.
    """
    text = element.text
    if text is None:
        raise HPSParseError(f"Element '{element.tag}' has no binary data")

    return base64.b64decode(text.strip())


def should_scramble_key(element: ET.Element) -> bool:
    """Determine if the encryption key should be scrambled based on XML attributes.

    :param element: The XML element containing the attribute.
    :return: Whether to scramble the key.
    """
    return element.get("Key") is not None


def extract_original_size(element: ET.Element, size_attribute_name: str = "Base64EncodedBytes") -> int | None:
    """Extract the original size of data from an XML element attribute.

    :param element: The XML element containing the attribute.
    :param size_attribute_name: The name of the attribute that contains the original size.
    :return: The original size as an integer, or None if not present.
    """
    if original_size_attr := element.get(size_attribute_name):
        return int(original_size_attr)

    return None


def extract_encrypted_data(element: ET.Element, size_attribute_name: str = "Base64EncodedBytes") -> EncryptedData:
    """Extract encrypted data from an XML element with metadata.

    :param element: The XML element containing encrypted data.
    :param size_attribute_name: The name of the attribute that contains the original size.
    :return: An EncryptedData object containing the data and metadata.
    """
    data = decode_binary_element(element)
    original_size = extract_original_size(element, size_attribute_name)
    use_scrambled_key = should_scramble_key(element)

    return EncryptedData(data, original_size, use_scrambled_key)


def extract_binary_data(
    element: ET.Element,
    is_encrypted: bool,
    size_attribute_name: str = "Base64EncodedBytes",
) -> bytes | EncryptedData:
    """Extract binary data from an XML element, handling encryption if necessary.

    :param element: The XML element containing the binary data.
    :param is_encrypted: Whether the data is encrypted.
    :param size_attribute_name: The name of the attribute that contains the original size.
    :return: The binary data as bytes or an EncryptedData object.
    """
    if is_encrypted:
        return extract_encrypted_data(element, size_attribute_name)

    return decode_binary_element(element)


def get_required_child(parent: ET.Element, path: str) -> ET.Element:
    """Get a required child element from an XML parent.

    :param parent: The parent XML element.
    :param path: The path to the child element.
    :return: The child XML element.
    :raises HPSParseError: If the child element is not found.
    """
    child = parent.find(path)
    if child is None:
        raise HPSParseError(f"Required XML element '{path}' not found.")

    return child


def get_required_text(element: ET.Element) -> str:
    """Get the text content of a required XML element.

    :param element: The XML element.
    :return: The text content.
    :raises HPSParseError: If the text content is missing.
    """
    text = element.text
    if text is None:
        raise HPSParseError(f"Element '{element.tag}' has no text content.")

    return text


def get_property_value(element: ET.Element, property_name: str) -> str:
    """Get the value attribute from a 'Property' element.

    :param element: The XML element.
    :param property_name: The name of the property to find.
    :return: The property value.
    :raises HPSParseError: If the property or its value is missing.
    """
    property_elem = element.find(f".//Property[@name='{property_name}']")
    if property_elem is None:
        raise HPSParseError(f"Missing 'Property' element with name='{property_name}'")

    value = property_elem.get("value")
    if value is None:
        raise HPSParseError(f"Missing 'value' attribute on Property[@name='{property_name}']")

    return value


def extract_control_points_packed(data: bytes) -> npt.NDArray[np.floating]:
    """Extract 3D control points from packed binary data (base64-encoded).

    :param data: The binary data containing packed float coordinates.
    :return: A numpy array of shape (N, 3) containing the control points.
    :raises HPSParseError: If the data length is not divisible by 4 (size of float) or if no valid points are found.
    """
    if len(data) % 4 != 0:
        raise HPSParseError(f"Packed control points data length {len(data)} is not divisible by 4 (sizeof float)")

    floats = np.frombuffer(data, dtype=np.float32)
    num_points = len(floats) // 3
    if num_points == 0:
        raise HPSParseError("No complete control points found in packed data")

    return floats[: num_points * 3].reshape(num_points, 3)


def extract_control_points_xml(element: ET.Element) -> npt.NDArray[np.floating]:
    """Extract 3D control points from XML elements.

    :param element: The XML element containing the control point objects.
    :return: A numpy array of shape (N, 3) containing the control points.
    :raises HPSParseError: If any control point is missing required attributes or if no valid points are found.
    """
    points: list[tuple[float, float, float]] = []

    for obj in element.findall("Object"):
        vector = obj.find("Vector[@name='p']")
        if vector is None:
            raise HPSParseError("Object in ControlPoints is missing Vector[@name='p'] element")

        x_str = vector.get("x")
        y_str = vector.get("y")
        z_str = vector.get("z")
        if x_str is None or y_str is None or z_str is None:
            raise HPSParseError("Vector element is missing x, y, or z attribute")

        try:
            x = float(x_str)
            y = float(y_str)
            z = float(z_str)
        except ValueError as e:
            raise HPSParseError(f"Failed to parse vector coordinates: {e}") from e

        points.append((x, y, z))

    if not points:
        raise HPSParseError("ControlPoints element contains no valid control points")

    return np.array(points, dtype=np.float32)


def parse_spline(element: ET.Element) -> Spline:
    """Parse a single spline from an XML Object element.

    :param element: The XML element representing the spline object.
    :return: A Spline object containing the parsed data.
    :raises HPSParseError: If required elements or attributes are missing.
    """
    name = get_property_value(element, "Name")
    radius_str = get_property_value(element, "Radius")
    closed_str = get_property_value(element, "Closed")
    color_str = get_property_value(element, "Color")
    misc_str = get_property_value(element, "iMisc1")

    try:
        radius = float(radius_str)
        is_cyclic = closed_str.lower() == "true"
        color = int(color_str)
        misc = int(misc_str)
    except ValueError as e:
        raise HPSParseError(f"Failed to parse spline property values: {e}") from e

    control_points_packed_elem = element.find(".//ControlPointsPacked")
    control_points_xml_elem = element.find(".//ControlPoints")

    if control_points_packed_elem is not None:
        control_points_text = control_points_packed_elem.text
        if control_points_text is None or control_points_text.strip() == "":
            raise HPSParseError("ControlPointsPacked element has no content")

        control_points_data = base64.b64decode(control_points_text.strip())
        control_points = extract_control_points_packed(control_points_data)
    elif control_points_xml_elem is not None:
        control_points = extract_control_points_xml(control_points_xml_elem)
    else:
        raise HPSParseError("Spline object is missing control points")

    return Spline(
        name=name,
        control_points=control_points,
        radius=radius,
        is_cyclic=is_cyclic,
        color=color,
        misc=misc,
    )


def parse_splines(root: ET.Element) -> list[Spline]:
    """Parse all splines from the XML root element.

    :param root: The root XML element of the HPS file.
    :return: A list of parsed Spline objects.
    """
    splines: list[Spline] = []

    splines_container = root.find(".//Splines")
    if splines_container is None:
        return splines

    for spline_obj in splines_container.findall(".//Object[@name='Spline']"):
        spline = parse_spline(spline_obj)
        splines.append(spline)

    return splines


def parse_xml(file: str | os.PathLike[str] | t.IO[bytes] | bytes) -> ET.ElementTree:
    """Parse an HPS XML file.

    :param file: The path to the HPS file, raw bytes, or a file-like object.
    :return: The parsed XML tree.
    """
    return ET.parse(file)


def load_hps(
    file: str | os.PathLike[str] | t.IO[bytes] | bytes,
    encryption_key: bytes | EncryptionKeyProvider | None = None,
) -> tuple[HPSPackedScan, HPSMesh]:
    """Load an HPS file and decode its contents.

    :param file: The path to the HPS file, raw bytes, or a file-like object.
    :param encryption_key: The encryption key for encrypted schemas. Can be raw bytes,
        an :py:class:`hpsdecode.encryption.EncryptionKeyProvider`, or ``None`` to read the key
        from the ``HPS_ENCRYPTION_KEY`` environment variable.
    :return: A tuple containing the packed scan metadata and the decoded mesh.
    :raises HPSSchemaError: If the file uses an unsupported compression schema.
    :raises HPSParseError: If the file structure is invalid.
    :raises HPSEncryptionError: If decryption fails (CE schema only).

    .. code-block:: python

        # Unencrypted file
        packed, mesh = load_hps("model.hps")

        # Encrypted file with static key
        packed, mesh = load_hps("encrypted.hps", encryption_key=bytes([28, 141, 16, ...]))

        # Encrypted file with custom provider
        packed, mesh = load_hps("encrypted.hps", encryption_key=MyKeyProvider())

        # Encrypted file using environment variable (set 'HPS_ENCRYPTION_KEY')
        packed, mesh = load_hps("encrypted.hps")

    """
    tree = parse_xml(file)
    root = tree.getroot()

    schema: SchemaType = get_required_text(get_required_child(root, ".//Schema"))  # type: ignore[assignment]
    if schema not in SUPPORTED_SCHEMAS:
        raise HPSSchemaError(schema, SUPPORTED_SCHEMAS)

    is_encrypted = schema in ("CE",)

    data_element = get_required_child(root, f".//{schema}")
    vertices_element = get_required_child(data_element, ".//Vertices")
    faces_element = get_required_child(data_element, ".//Facets")

    vertex_data = extract_binary_data(vertices_element, is_encrypted, "base64_encoded_bytes")
    face_data = decode_binary_element(faces_element)

    num_vertices = int(vertices_element.get("vertex_count", "0"))
    num_faces = int(faces_element.get("facet_count", "0"))
    check_value = vertices_element.get("check_value")
    default_vertex_color = vertices_element.get("color")
    default_face_color = faces_element.get("color")

    vertex_colors_data: bytes | EncryptedData | None = None
    vertex_colors_element = root.find(".//VertexColorSets/VertexColorSet")
    if vertex_colors_element is not None:
        vertex_colors_data = extract_binary_data(vertex_colors_element, is_encrypted)

    texture_coords_data: bytes | EncryptedData | None = None
    texture_coords_element = root.find(".//PerVertexTextureCoord")
    if texture_coords_element is not None:
        texture_coords_data = extract_binary_data(texture_coords_element, is_encrypted)

    texture_images: dict[str, bytes | EncryptedData] = {}
    seen_elements: set[int] = set()

    texture_paths = [
        *((p, True) for p in TEXTURE_IMAGE_PATHS_ENCRYPTED),
        *((p, False) for p in TEXTURE_IMAGE_PATHS_UNENCRYPTED),
    ]

    for path, encryptable in texture_paths:
        for texture_image_element in root.findall(path):
            element_id = id(texture_image_element)
            if element_id in seen_elements:
                continue

            seen_elements.add(element_id)
            texture_images[f"{path}:{element_id}"] = extract_binary_data(
                element=texture_image_element,
                is_encrypted=is_encrypted and encryptable,
            )

    splines = parse_splines(root)

    properties: dict[str, t.Any] = {}
    properties_element = root.find("Properties")
    if properties_element is not None:
        for property in properties_element.findall("Property"):
            name = property.get("name")
            value = property.get("value")

            if name is not None and value is not None:
                properties[name] = value

    context = ParseContext(
        vertex_data=vertex_data,
        face_data=face_data,
        vertex_count=num_vertices,
        face_count=num_faces,
        default_vertex_color=int(default_vertex_color) if default_vertex_color else None,
        default_face_color=int(default_face_color) if default_face_color else None,
        vertex_colors_data=vertex_colors_data,
        texture_coords_data=texture_coords_data,
        texture_images=list(texture_images.values()),
        splines=splines,
        check_value=int(check_value) if check_value else None,
        properties=properties,
    )

    parser = get_parser(schema, encryption_key)
    result = parser.parse(context)

    if result.mesh.num_vertices != num_vertices:
        raise HPSParseError(f"Vertex count mismatch: expected {num_vertices}, got {result.mesh.num_vertices}")

    if result.mesh.num_faces != num_faces:
        raise HPSParseError(f"Face count mismatch: expected {num_faces}, got {result.mesh.num_faces}")

    packed = HPSPackedScan(
        schema=schema,
        num_vertices=num_vertices,
        num_faces=num_faces,
        vertex_data=context.vertex_data,
        face_data=context.face_data,
        default_vertex_color=context.default_vertex_color,
        default_face_color=context.default_face_color,
        vertex_colors_data=context.vertex_colors_data,
        texture_coords_data=context.texture_coords_data,
        texture_images=context.texture_images,
        splines=context.splines,
        vertex_commands=result.vertex_commands,
        face_commands=result.face_commands,
        check_value=context.check_value,
        properties=context.properties,
    )

    return packed, result.mesh
