"""Utilities for parsing and decoding HIMSA packed standard (HPS) files."""

from __future__ import annotations

__all__ = ["load_hps"]

import base64
import typing as t
import xml.etree.ElementTree as ET

import numpy as np

from hpsdecode.exceptions import HPSParseError, HPSSchemaError
from hpsdecode.mesh import HPSMesh, HPSPackedScan, SchemaType
from hpsdecode.schemas import SUPPORTED_SCHEMAS, get_parser

if t.TYPE_CHECKING:
    import os


def decode_binary_element(element: ET.Element) -> bytes:
    """Decode base64-encoded binary data from an XML element.

    :param element: The XML element containing base64-encoded data.
    :return: The decoded binary data.
    """
    expected_length = int(element.get("base64_encoded_bytes", "0"))
    text = element.text
    if text is None:
        raise HPSParseError(f"Element '{element.tag}' has no binary data")

    data = base64.b64decode(text.strip())
    if len(data) != expected_length:
        raise HPSParseError(
            f"Binary data length mismatch in '{element.tag}': expected {expected_length}, got {len(data)}"
        )

    return data


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


def load_hps(file: str | os.PathLike[str] | bytes) -> tuple[HPSPackedScan, HPSMesh]:
    """Load an HPS file and decode its contents.

    :param file: The path to the HPS file, raw bytes, or a file-like object.
    :return: A tuple containing the packed scan metadata and the decoded mesh.
    :raises HPSSchemaError: If the file uses an unsupported compression schema.
    :raises HPSParseError: If the file structure is invalid.

    .. code-block:: python

        packed, mesh = load_hps("model.hps")
        print(f"Schema: {packed.schema}")
        print(f"Loaded {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")

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

    parser = get_parser(schema)
    result = parser.parse(vertex_data, face_data)

    if result.mesh.num_vertices != num_vertices:
        raise HPSParseError(f"Vertex count mismatch: expected {num_vertices}, got {result.mesh.num_vertices}")

    if result.mesh.num_faces != num_faces:
        raise HPSParseError(f"Face count mismatch: expected {num_faces}, got {result.mesh.num_faces}")

    if result.mesh.face_colors.size == 0 and (color := faces_element.get("color")):
        color = int(color)
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF

        result.mesh.face_colors = np.tile(np.array([[r, g, b]], dtype=np.uint8), (num_faces, 1))

    packed = HPSPackedScan(
        schema=schema,
        num_vertices=num_vertices,
        num_faces=num_faces,
        vertex_commands=result.vertex_commands,
        face_commands=result.face_commands,
    )

    return packed, result.mesh
