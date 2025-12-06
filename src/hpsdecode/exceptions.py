"""Exceptions for HPS decoding errors."""

__all__ = ["HPSEncryptionError", "HPSParseError", "HPSSchemaError"]


class HPSEncryptionError(Exception):
    """Raised when decryption of HPS data fails."""

    pass


class HPSParseError(Exception):
    """Raised when parsing binary data fails."""

    def __init__(self, message: str, *, offset: int | None = None) -> None:
        """Initialize the parse error.

        :param message: The description of the parsing failure.
        :param offset: The byte offset where the error occurred, if known.
        """
        self.offset = offset
        full_message = f"{message} (at offset 0x{offset:X})" if offset is not None else message
        super().__init__(full_message)


class HPSSchemaError(Exception):
    """Raised when an unsupported or invalid schema is encountered."""

    def __init__(self, schema: str, supported: tuple[str, ...]) -> None:
        """Initialize the schema error.

        :param schema: The unsupported schema identifier.
        :param supported: A tuple of supported schema identifiers.
        """
        self.schema = schema
        self.supported = supported
        super().__init__(f"Unsupported schema '{schema}'. Supported: {', '.join(supported)}")
