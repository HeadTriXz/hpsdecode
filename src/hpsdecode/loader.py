"""Utilities for parsing and decoding HIMSA packed standard (HPS) files."""

from __future__ import annotations

__all__ = ["load_hps"]

import base64
import typing as t
import xml.etree.ElementTree as ET

from hpsdecode.exceptions import HPSParseError, HPSSchemaError
from hpsdecode.mesh import HPSMesh, HPSPackedScan, SchemaType
from hpsdecode.schemas import SUPPORTED_SCHEMAS, ParseContext, get_parser

if t.TYPE_CHECKING:
    import os

    from hpsdecode.encryption import EncryptionKeyProvider


def decode_binary_element(element: ET.Element) -> bytes:
    """Decode base64-encoded binary data from an XML element.

    :param element: The XML element containing base64-encoded data.
    :return: The decoded binary data.
    """
    text = element.text
    if text is None:
        raise HPSParseError(f"Element '{element.tag}' has no binary data")

    return base64.b64decode(text.strip())


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


def parse_xml(file: str | os.PathLike[str] | bytes) -> ET.ElementTree:
    """Parse an HPS XML file.

    :param file: The path to the HPS file, raw bytes, or a file-like object.
    :return: The parsed XML tree.
    """
    return ET.parse(file)


def load_hps(
    file: str | os.PathLike[str] | bytes,
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

    data_element = get_required_child(root, f".//{schema}")
    vertices_element = get_required_child(data_element, ".//Vertices")
    faces_element = get_required_child(data_element, ".//Facets")

    vertex_data = decode_binary_element(vertices_element)
    face_data = decode_binary_element(faces_element)

    num_vertices = int(vertices_element.get("vertex_count", "0"))
    num_faces = int(faces_element.get("facet_count", "0"))
    check_value = vertices_element.get("check_value")
    default_vertex_color = vertices_element.get("color")
    default_face_color = faces_element.get("color")

    context = ParseContext(
        vertex_data=vertex_data,
        face_data=face_data,
        vertex_count=num_vertices,
        face_count=num_faces,
        default_vertex_color=int(default_vertex_color) if default_vertex_color else None,
        default_face_color=int(default_face_color) if default_face_color else None,
        check_value=int(check_value) if check_value else None,
    )

    properties_element = root.find("Properties")
    if properties_element is not None:
        for property in properties_element.findall("Property"):
            name = property.get("name")
            value = property.get("value")

            if name is not None and value is not None:
                context.properties[name] = value

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
        vertex_data=vertex_data,
        face_data=face_data,
        vertex_commands=result.vertex_commands,
        face_commands=result.face_commands,
    )

    return packed, result.mesh
