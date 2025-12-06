"""Schema-specific parsers for HPS compression formats."""

from __future__ import annotations

__all__ = [
    "SUPPORTED_SCHEMAS",
    "BaseSchemaParser",
    "CASchemaParser",
    "CBSchemaParser",
    "CCSchemaParser",
    "CESchemaParser",
    "ParseContext",
    "ParseResult",
    "get_parser",
]

import typing as t

from hpsdecode.schemas.base import BaseSchemaParser, ParseContext, ParseResult
from hpsdecode.schemas.ca import CASchemaParser
from hpsdecode.schemas.cb import CBSchemaParser
from hpsdecode.schemas.cc import CCSchemaParser
from hpsdecode.schemas.ce import CESchemaParser

if t.TYPE_CHECKING:
    from hpsdecode.mesh import SchemaType


#: Tuple of all supported HPS compression schemas.
SUPPORTED_SCHEMAS: t.Final[tuple[SchemaType, ...]] = ("CA", "CB", "CC", "CE")

#: Mapping of schema identifiers to their respective parser classes.
_SCHEMA_PARSERS: t.Final[t.Dict[SchemaType, t.Type[BaseSchemaParser]]] = {
    "CA": CASchemaParser,
    "CB": CBSchemaParser,
    "CC": CCSchemaParser,
    "CE": CESchemaParser,
}


def get_parser(schema: SchemaType) -> BaseSchemaParser:
    """Get the parser for the specified HPS compression schema.

    :param schema: The schema identifier (e.g., "CA", "CB", "CC", "CE").
    :return: An instance of the corresponding schema parser.
    :raises KeyError: If the schema is not supported.
    """
    parser_class = _SCHEMA_PARSERS[schema]
    return parser_class()
