"""Base schema parser definitions for HPS decoding."""

from __future__ import annotations

__all__ = ["BaseSchemaParser", "ParseResult"]

import abc
import dataclasses
import typing as t

if t.TYPE_CHECKING:
    import hpsdecode.commands as hpc
    from hpsdecode.mesh import HPSMesh


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
    def parse(self, vertex_data: bytes, face_data: bytes) -> ParseResult:
        """Parse HPS binary data for the specific schema.

        :param vertex_data: The binary data for vertex commands.
        :param face_data: The binary data for face commands.
        :return: The parsing result containing the decoded mesh and commands.
        """
        pass
