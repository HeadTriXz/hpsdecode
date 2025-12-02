"""HPS command definitions for vertex and face operations."""

__all__ = [
    "VertexCommandType",
    "FaceCommandType",
    "VertexCommand",
    "FaceCommand",
    "UseRelativeCoordinates",
    "UseAbsoluteCoordinates",
    "DisableTextureCoordinates",
    "EnableTextureCoordinates",
    "SetTextureImage",
    "SetBitsPerTextureCoordinate",
    "SetMultiplier",
    "SetColor",
    "VertexList",
    "Previous",
    "Next",
    "Ignore",
    "Restart",
    "Restart16",
    "Restart32",
    "Absolute16",
    "Absolute32",
    "Remove",
    "IncreaseVertexListPointer",
    "AnyVertexCommand",
    "AnyFaceCommand",
]

import dataclasses
import enum
import typing as t


class VertexCommandType(enum.IntEnum):
    """Enumeration of all vertex command types in HPS 501."""

    USE_RELATIVE_COORDINATES = 1
    USE_ABSOLUTE_COORDINATES = 2
    DISABLE_TEXTURE_COORDINATES = 10
    ENABLE_TEXTURE_COORDINATES = 11
    SET_TEXTURE_IMAGE = 12
    SET_BITS_PER_TEXTURE_COORDINATE = 13
    SET_MULTIPLIER = 20
    SET_COLOR = 30


class FaceCommandType(enum.IntEnum):
    """Enumeration of all face/facet command types in HPS 501."""

    VERTEX_LIST = 0
    PREVIOUS = 1
    NEXT = 2
    IGNORE = 3
    RESTART = 4
    RESTART_16 = 5
    RESTART_32 = 6
    ABSOLUTE_16 = 7
    ABSOLUTE_32 = 8
    REMOVE = 9
    INCREASE_VERTEX_LIST_POINTER = 10


class VertexCommand:
    """Base class for vertex commands."""

    #: The type of vertex command
    op: VertexCommandType


class FaceCommand:
    """Base class for face/facet commands."""

    #: The type of face command
    op: FaceCommandType


@dataclasses.dataclass
class UseRelativeCoordinates(VertexCommand):
    """Use relative vertex coordinate vectors."""

    #: The vertex command type.
    op: VertexCommandType = VertexCommandType.USE_RELATIVE_COORDINATES


@dataclasses.dataclass
class UseAbsoluteCoordinates(VertexCommand):
    """Use absolute vertex coordinate vectors."""

    #: The vertex command type.
    op: VertexCommandType = VertexCommandType.USE_ABSOLUTE_COORDINATES


@dataclasses.dataclass
class DisableTextureCoordinates(VertexCommand):
    """Vertex coordinates after this command cannot contain texture coordinates."""

    #: The vertex command type.
    op: VertexCommandType = VertexCommandType.DISABLE_TEXTURE_COORDINATES


@dataclasses.dataclass
class EnableTextureCoordinates(VertexCommand):
    """Vertex coordinates after this command must include texture coordinates."""

    #: The vertex command type.
    op: VertexCommandType = VertexCommandType.ENABLE_TEXTURE_COORDINATES


@dataclasses.dataclass
class SetTextureImage(VertexCommand):
    """Changes the current texture image."""

    #: ID of the texture image to use for subsequent vertices
    image_id: int

    #: The vertex command type.
    op: VertexCommandType = VertexCommandType.SET_TEXTURE_IMAGE


@dataclasses.dataclass
class SetBitsPerTextureCoordinate(VertexCommand):
    """Texture coordinates after this command must use a specified number of bits per coordinate."""

    #: Number of bits to use per texture coordinate
    bits: int

    #: The vertex command type.
    op: VertexCommandType = VertexCommandType.SET_BITS_PER_TEXTURE_COORDINATE


@dataclasses.dataclass
class SetMultiplier(VertexCommand):
    """Sets the multiplier for vertex coordinates."""

    #: Multiplier factor applied to integer vertex coordinates.
    multiplier: float

    #: The vertex command type.
    op: VertexCommandType = VertexCommandType.SET_MULTIPLIER


@dataclasses.dataclass
class SetColor(VertexCommand):
    """Vertices after this command must use the specified RGB color."""

    #: The red component (0-255).
    r: int

    #: The green component (0-255).
    g: int

    #: The blue component (0-255).
    b: int

    #: The vertex command type.
    op: VertexCommandType = VertexCommandType.SET_COLOR


@dataclasses.dataclass
class VertexList(FaceCommand):
    """Create new face from current edge, using the next vertex in the global vertex list."""

    #: The face command type.
    op: FaceCommandType = FaceCommandType.VERTEX_LIST


@dataclasses.dataclass
class Previous(FaceCommand):
    """Create new facet from current edge, using the vertex at the beginning of the previous edge."""

    #: The face command type.
    op: FaceCommandType = FaceCommandType.PREVIOUS


@dataclasses.dataclass
class Next(FaceCommand):
    """Create new facet from current edge, using the vertex at the end of the next edge."""

    #: The face command type.
    op: FaceCommandType = FaceCommandType.NEXT


@dataclasses.dataclass
class Ignore(FaceCommand):
    """Ignore current edge, and assign current edge to the next edge in the edge list."""

    #: The face command type.
    op: FaceCommandType = FaceCommandType.IGNORE


@dataclasses.dataclass
class Restart(FaceCommand):
    """Create new facet using three next vertices in the global list."""

    #: The face command type.
    op: FaceCommandType = FaceCommandType.RESTART


@dataclasses.dataclass
class Restart16(FaceCommand):
    """Create new facet using three specified vertices, with 16-bit indices."""

    #: Index of the first vertex (16-bit)
    v0: int

    #: Index of the second vertex (16-bit)
    v1: int

    #: Index of the third vertex (16-bit)
    v2: int

    #: The face command type.
    op: FaceCommandType = FaceCommandType.RESTART_16


@dataclasses.dataclass
class Restart32(FaceCommand):
    """Create new facet using three specified vertices, with 32-bit indices."""

    #: Index of the first vertex (32-bit)
    v0: int

    #: Index of the second vertex (32-bit)
    v1: int

    #: Index of the third vertex (32-bit)
    v2: int

    #: The face command type.
    op: FaceCommandType = FaceCommandType.RESTART_32


@dataclasses.dataclass
class Absolute16(FaceCommand):
    """Create new facet from current edge using specified vertex, with 16-bit index."""

    #: Index of the vertex (16-bit)
    v: int

    #: The face command type.
    op: FaceCommandType = FaceCommandType.ABSOLUTE_16


@dataclasses.dataclass
class Absolute32(FaceCommand):
    """Create new facet from current edge using specified vertex, with 32-bit index."""

    #: Index of the vertex (32-bit)
    v: int

    #: The face command type.
    op: FaceCommandType = FaceCommandType.ABSOLUTE_32


@dataclasses.dataclass
class Remove(FaceCommand):
    """Remove edge from the current edge list, and assign current edge to the next edge in the list."""

    #: The face command type.
    op: FaceCommandType = FaceCommandType.REMOVE


@dataclasses.dataclass
class IncreaseVertexListPointer(FaceCommand):
    """Increase the vertex list pointer by one without creating a face."""

    #: The face command type.
    op: FaceCommandType = FaceCommandType.INCREASE_VERTEX_LIST_POINTER


AnyVertexCommand: t.TypeAlias = (
    UseRelativeCoordinates
    | UseAbsoluteCoordinates
    | DisableTextureCoordinates
    | EnableTextureCoordinates
    | SetTextureImage
    | SetBitsPerTextureCoordinate
    | SetMultiplier
    | SetColor
)

AnyFaceCommand: t.TypeAlias = (
    VertexList
    | Previous
    | Next
    | Ignore
    | Restart
    | Restart16
    | Restart32
    | Absolute16
    | Absolute32
    | Remove
    | IncreaseVertexListPointer
)
