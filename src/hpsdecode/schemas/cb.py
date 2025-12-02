"""Parser for the 'CB' HPS compression schema."""

__all__ = ["CBSchemaParser"]

from hpsdecode.schemas.base import BaseSchemaParser, ParseResult


class CBSchemaParser(BaseSchemaParser):
    """Parser for the 'CB' HPS compression schema."""

    def parse(self, vertex_data: bytes, face_data: bytes) -> ParseResult:
        """Parse HPS binary data for the 'CB' schema.

        :param vertex_data: The binary data for vertex commands.
        :param face_data: The binary data for face commands.
        :return: The parsing result containing the decoded mesh and commands.
        """
        raise NotImplementedError()
