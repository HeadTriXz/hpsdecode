"""Base schema parser definitions for HPS decoding."""

from __future__ import annotations

__all__ = ["BaseSchemaParser", "EncryptedData", "ParseContext", "ParseResult"]

import abc
import dataclasses
import typing as t

if t.TYPE_CHECKING:
    import hpsdecode.commands as hpc
    from hpsdecode.mesh import HPSMesh


@dataclasses.dataclass(frozen=True)
class EncryptedData:
    """Container for encrypted data with metadata needed for decryption."""

    #: The encrypted binary data.
    data: bytes

    #: The original size before padding (from XML attributes).
    original_size: int | None = None

    #: Whether this data should be decrypted with a scrambled key.
    use_scrambled_key: bool = False


@dataclasses.dataclass(frozen=True)
class ParseContext:
    """Context object containing metadata required for parsing HPS data."""

    #: The binary data for the vertices.
    vertex_data: EncryptedData | bytes | None = None

    #: The binary data for the faces.
    face_data: bytes | None = None

    #: The number of vertices in the mesh, if known.
    vertex_count: int | None = None

    #: The number of faces in the mesh, if known.
    face_count: int | None = None

    #: The default vertex color to use if it cannot be determined from the data.
    default_vertex_color: int | None = None

    #: The default face color to use if it cannot be determined from the data.
    default_face_color: int | None = None

    #: The vertex color data.
    vertex_colors_data: EncryptedData | bytes | None = None

    #: The texture coordinate data.
    texture_coords_data: EncryptedData | bytes | None = None

    #: A list of texture images.
    texture_images: list[EncryptedData | bytes] = dataclasses.field(default_factory=list)

    #: The value used for integrity checking, if available.
    check_value: int | None = None

    #: The HPS file properties (e.g., encryption keys, package locks).
    properties: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ParseResult:
    """Result of parsing HPS binary data."""

    #: The decoded mesh.
    mesh: HPSMesh

    #: The sequence of vertex commands.
    vertex_commands: list[hpc.AnyVertexCommand]

    #: The sequence of face commands.
    face_commands: list[hpc.AnyFaceCommand]


class BaseSchemaParser(abc.ABC):
    """Abstract base class for HPS schema parsers."""

    @abc.abstractmethod
    def parse(self, context: ParseContext) -> ParseResult:
        """Parse HPS binary data for the specific schema.

        :param context: The parsing context containing metadata and binary data.
        :return: The parsing result containing the decoded mesh and commands.
        """
        pass
