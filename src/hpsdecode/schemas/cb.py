"""Parser for the 'CB' HPS compression schema."""

__all__ = ["CBSchemaParser"]

from hpsdecode.schemas.base import BaseSchemaParser, ParseContext, ParseResult


class CBSchemaParser(BaseSchemaParser):
    """Parser for the 'CB' HPS compression schema."""

    def parse(self, context: ParseContext) -> ParseResult:
        """Parse HPS binary data for the 'CB' schema.

        :param context: The parsing context containing metadata and binary data.
        :return: The parsing result containing the decoded mesh and commands.
        """
        raise NotImplementedError()
