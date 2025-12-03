"""Parser for the 'CC' HPS compression schema."""

from __future__ import annotations

__all__ = ["CCSchemaParser"]

import typing as t

import numpy as np

import hpsdecode.commands as hpc
from hpsdecode.binary import BinaryReader
from hpsdecode.exceptions import HPSParseError
from hpsdecode.mesh import Edge, HPSMesh
from hpsdecode.schemas import BaseSchemaParser, ParseResult

if t.TYPE_CHECKING:
    import numpy.typing as npt


class CCSchemaParser(BaseSchemaParser):
    """Parser for the 'CC' HPS compression schema."""

    _current_edge_idx: int
    _global_vertex_ptr: int
    _edge_list: list[Edge]
    _faces: list[tuple[int, int, int]]

    def __init__(self) -> None:
        """Initialize the CC schema parser."""
        self._clear()

    def parse(self, vertex_data: bytes, face_data: bytes) -> ParseResult:
        """Parse HPS binary data for the 'CC' schema.

        :param vertex_data: The binary data for vertex commands.
        :param face_data: The binary data for face commands.
        :return: The parsing result containing the decoded mesh and commands.
        """
        vertices, vertex_commands = self.parse_vertices(vertex_data)
        faces, face_commands = self.parse_faces(face_data)

        return ParseResult(
            mesh=HPSMesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=np.empty((0, 3), dtype=np.uint8),
                face_colors=np.empty((0, 3), dtype=np.uint8),
                uv=np.empty((0, 2), dtype=np.float32),
            ),
            vertex_commands=vertex_commands,
            face_commands=face_commands,
        )

    def parse_faces(self, data: bytes) -> tuple[npt.NDArray[np.integer], list[hpc.AnyFaceCommand]]:
        """Parse face data from bytes.

        :param data: The raw byte data containing face data.
        :return: An array of face indices (M, 3) and the face commands.
        """
        self._clear()

        commands = self._parse_commands(data)
        for command in commands:
            self._process_command(command)

        faces = np.array(self._faces, dtype=np.int32)
        return faces, commands

    def parse_vertices(self, data: bytes) -> tuple[npt.NDArray[np.floating], list[hpc.AnyVertexCommand]]:
        """Parse vertex data from bytes.

        :param data: The raw byte data containing vertex data.
        :return: An array of vertex positions (N, 3) and the vertex commands.
        """
        vertices = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
        commands = []

        return vertices, commands

    def _clear(self) -> None:
        """Reset the internal parser state for a new mesh."""
        self._current_edge_idx = 0
        self._global_vertex_ptr = 0
        self._edge_list = []
        self._faces = []

    def _create_restart_face(self, v0: int, v1: int, v2: int) -> None:
        """Create a new face and reset the edge list.

        :param v0: The first vertex index of the new face.
        :param v1: The second vertex index of the new face.
        :param v2: The third vertex index of the new face.
        """
        self._faces.append((v0, v1, v2))

        self._edge_list = [
            Edge(v0, v1),
            Edge(v1, v2),
            Edge(v2, v0),
        ]
        self._current_edge_idx = 0

    def _extend_current_edge(self, v: int) -> None:
        """Create a new face by extending the current edge to a new vertex.

        :param v: The new vertex index to extend the edge to.
        """
        if not self._edge_list:
            raise HPSParseError("No edges available to extend")

        current_edge = self._edge_list[self._current_edge_idx]

        self._faces.append((v, current_edge.end, current_edge.start))

        new_edge1 = Edge(current_edge.start, v)
        new_edge2 = Edge(v, current_edge.end)

        self._edge_list.pop(self._current_edge_idx)
        self._edge_list.insert(self._current_edge_idx, new_edge2)
        self._edge_list.insert(self._current_edge_idx, new_edge1)


    def _handle_previous(self) -> None:
        """Create a face using the previous edge's start vertex."""
        if len(self._edge_list) < 2:
            raise HPSParseError("Cannot perform Previous with fewer than 2 edges")

        n = len(self._edge_list)
        prev_idx = (self._current_edge_idx - 1 + n) % n
        curr_idx = self._current_edge_idx

        prev_edge = self._edge_list[prev_idx]
        curr_edge = self._edge_list[curr_idx]

        self._faces.append((curr_edge.start, prev_edge.start, curr_edge.end))

        new_edge = Edge(prev_edge.start, curr_edge.end)

        high_idx, low_idx = (curr_idx, prev_idx) if curr_idx > prev_idx else (prev_idx, curr_idx)
        self._edge_list.pop(high_idx)
        self._edge_list.pop(low_idx)
        self._edge_list.insert(low_idx, new_edge)

        self._current_edge_idx = (low_idx + 1) % len(self._edge_list)

    def _handle_next(self) -> None:
        """Create a face using the next edge's end vertex."""
        if len(self._edge_list) < 2:
            raise HPSParseError("Cannot perform Next with fewer than 2 edges")

        curr_idx = self._current_edge_idx
        next_idx = (curr_idx + 1) % len(self._edge_list)

        curr_edge = self._edge_list[curr_idx]
        next_edge = self._edge_list[next_idx]

        self._faces.append((curr_edge.start, next_edge.end, curr_edge.end))

        new_edge = Edge(curr_edge.start, next_edge.end)

        high_idx, low_idx = (next_idx, curr_idx) if next_idx > curr_idx else (curr_idx, next_idx)
        self._edge_list.pop(high_idx)
        self._edge_list.pop(low_idx)
        self._edge_list.insert(low_idx, new_edge)

        self._current_edge_idx = (low_idx + 1) % len(self._edge_list)

    def _increase_edge_pointer(self, n: int = 1) -> None:
        """Increase the current edge pointer by n.

        :param n: The number of edges to advance the pointer by.
        """
        self._current_edge_idx = (self._current_edge_idx + n) % len(self._edge_list)

    def _next_global_vertex(self) -> int:
        """Get the next global vertex index and increment the pointer.

        :return: The next global vertex index.
        """
        v = self._global_vertex_ptr
        self._global_vertex_ptr += 1
        return v

    def _parse_commands(self, data: bytes) -> list[hpc.AnyFaceCommand]:
        """Parse face commands from the binary data.

        :param data: The raw byte data containing face commands.
        :return: A list of parsed face commands.
        """
        reader = BinaryReader(data)
        commands: list[hpc.AnyFaceCommand] = []

        while not reader.is_eof():
            try:
                command_byte = reader.read_uint8()
            except EOFError:
                break

            if command_byte >> 4 != 0:
                raise HPSParseError("Upper 4 bits of face command byte must be zero", offset=reader.position - 1)

            opcode = command_byte & 0x0F
            command = self._parse_single_command(reader, opcode)
            commands.append(command)

        return commands

    def _parse_single_command(self, reader: BinaryReader, opcode: int) -> hpc.AnyFaceCommand:
        """Parse a single command given its opcode.

        :param reader: The binary reader positioned after the opcode byte.
        :param opcode: The command opcode.
        :return: The parsed command.
        :raises HPSParseError: If the opcode is unknown.
        """
        match opcode:
            case hpc.FaceCommandType.VERTEX_LIST:
                return hpc.VertexList()
            case hpc.FaceCommandType.PREVIOUS:
                return hpc.Previous()
            case hpc.FaceCommandType.NEXT:
                return hpc.Next()
            case hpc.FaceCommandType.IGNORE:
                return hpc.Ignore()
            case hpc.FaceCommandType.RESTART:
                return hpc.Restart()
            case hpc.FaceCommandType.RESTART_16:
                return hpc.Restart16(
                    v0=reader.read_uint16(),
                    v1=reader.read_uint16(),
                    v2=reader.read_uint16(),
                )
            case hpc.FaceCommandType.RESTART_32:
                return hpc.Restart32(
                    v0=reader.read_uint32(),
                    v1=reader.read_uint32(),
                    v2=reader.read_uint32(),
                )
            case hpc.FaceCommandType.ABSOLUTE_16:
                # Why can the Absolute16 command have 32-bit values? (╯°□°）╯︵ ┻━┻
                return hpc.Absolute16(v=reader.read_uint32())
            case hpc.FaceCommandType.ABSOLUTE_32:
                return hpc.Absolute32(v=reader.read_uint32())
            case hpc.FaceCommandType.REMOVE:
                return hpc.Remove()
            case hpc.FaceCommandType.INCREASE_VERTEX_LIST_POINTER:
                return hpc.IncreaseVertexListPointer()
            case _:
                raise HPSParseError(f"Unknown face command opcode: {opcode}", offset=reader.position)

    def _process_command(self, command: hpc.AnyFaceCommand) -> None:
        match command.op:
            case hpc.FaceCommandType.VERTEX_LIST:
                v = self._next_global_vertex()
                self._extend_current_edge(v)
                self._increase_edge_pointer(2)
            case hpc.FaceCommandType.PREVIOUS:
                self._handle_previous()
            case hpc.FaceCommandType.NEXT:
                self._handle_next()
            case hpc.FaceCommandType.IGNORE:
                self._increase_edge_pointer()
            case hpc.FaceCommandType.RESTART:
                v0 = self._next_global_vertex()
                v1 = self._next_global_vertex()
                v2 = self._next_global_vertex()
                self._create_restart_face(v0, v1, v2)
            case hpc.FaceCommandType.RESTART_16 | hpc.FaceCommandType.RESTART_32:
                self._create_restart_face(command.v0, command.v1, command.v2)
            case hpc.FaceCommandType.ABSOLUTE_16 | hpc.FaceCommandType.ABSOLUTE_32:
                self._extend_current_edge(command.v)
                self._increase_edge_pointer(2)
            case hpc.FaceCommandType.REMOVE:
                self._remove_current_edge()
            case hpc.FaceCommandType.INCREASE_VERTEX_LIST_POINTER:
                self._next_global_vertex()
            case _:
                raise HPSParseError(f"Cannot process unknown command: {command}")

    def _remove_current_edge(self) -> None:
        """Remove the current edge from the edge list."""
        if not self._edge_list:
            raise HPSParseError("Cannot remove edge: edge list is empty")

        n = len(self._edge_list)
        prev_idx = (self._current_edge_idx - 1 + n) % n
        curr_idx = self._current_edge_idx

        prev_edge = self._edge_list[prev_idx]
        curr_edge = self._edge_list[curr_idx]

        if prev_edge.start == curr_edge.end and n > 2:
            # Case A: Current edge turns back (remove both edges)
            high_idx, low_idx = (curr_idx, prev_idx) if curr_idx > prev_idx else (prev_idx, curr_idx)
            self._edge_list.pop(high_idx)
            self._edge_list.pop(low_idx)

            if self._edge_list:
                new_prev_idx = (low_idx - 1 + len(self._edge_list)) % len(self._edge_list)
                new_curr_idx = low_idx % len(self._edge_list)

                new_prev_edge = self._edge_list[new_prev_idx]
                new_curr_edge = self._edge_list[new_curr_idx]
                new_prev_edge.end = new_curr_edge.start

                self._current_edge_idx = new_curr_idx
            else:
                self._current_edge_idx = 0
        else:
            # Case B: Remove just current edge
            prev_edge.end = curr_edge.end
            self._edge_list.pop(curr_idx)
            if self._edge_list:
                self._current_edge_idx = curr_idx % len(self._edge_list)
            else:
                self._current_edge_idx = 0
