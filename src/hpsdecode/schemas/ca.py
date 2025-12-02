"""Parser for the 'CA' HPS compression schema."""

__all__ = ["CASchemaParser"]

from hpsdecode.schemas.cc import CCSchemaParser


class CASchemaParser(CCSchemaParser):
    """Parser for the 'CA' HPS compression schema.

    The CA compression schema is designed for backwards compatibility and
    uses the same format as the CC compression schema. This parser inherits
    all functionality from :py:class:`CCSchemaParser` without modification.
    """

    pass
