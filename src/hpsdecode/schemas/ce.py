"""Parser for the 'CE' HPS compression schema."""

from __future__ import annotations

__all__ = ["CESchemaParser"]

import hashlib

from hpsdecode.encryption import BlowfishDecryptor, EncryptionKeyProvider, EnvironmentKeyProvider, StaticKeyProvider
from hpsdecode.schemas.base import ParseContext, ParseResult
from hpsdecode.schemas.cc import CCSchemaParser


class CESchemaParser(CCSchemaParser):
    """Parser for the 'CE' HPS compression schema.

    The CE schema is CC with Blowfish encryption on vertex data.
    """

    HPS_ATTR_ENCRYPTION_KEY_ID: str = "EKID"
    HPS_ATTR_PACKAGE_LOCK_LIST: str = "PackageLockList"
    ENCRYPTION_KEY_ID_3SHAPE_INTERNAL: str = "1"

    _key_provider: EncryptionKeyProvider

    def __init__(self, encryption_key: bytes | EncryptionKeyProvider | None = None) -> None:
        """Initialize the CE schema parser.

        :param encryption_key: Encryption key as bytes, a key provider or None. If None, it will use
            the EnvironmentKeyProvider to read from the environment variable (``HPS_ENCRYPTION_KEY``).
        """
        super().__init__()

        if isinstance(encryption_key, bytes):
            self._key_provider = StaticKeyProvider(encryption_key)
        elif isinstance(encryption_key, EncryptionKeyProvider):
            self._key_provider = encryption_key
        else:
            self._key_provider = EnvironmentKeyProvider()

    def parse(self, context: ParseContext) -> ParseResult:
        """Parse encrypted HPS data for the 'CE' schema.

        :param context: The parsing context containing metadata and binary data.
        :return: The parsing result containing the decoded mesh and commands.
        """
        key = self._derive_key(context.properties)

        decrypted_vertex_data = BlowfishDecryptor(key).decrypt(context.vertex_data, context.vertex_count * 12)
        decrypted_context = ParseContext(
            properties=context.properties,
            vertex_data=decrypted_vertex_data,
            face_data=context.face_data,
            vertex_count=context.vertex_count,
            face_count=context.face_count,
            default_face_color=context.default_face_color,
        )

        return super().parse(decrypted_context)

    def _compute_package_lock_hash(self, properties: dict[str, str]) -> str | None:
        """Compute MD5 hash of the normalized package lock list.

        :param properties: HPS file properties.
        :return: Uppercase hex MD5 hash or None.
        """
        value = properties.get(self.HPS_ATTR_PACKAGE_LOCK_LIST)
        if not value:
            return None

        items = [item for item in value.split(";") if item]
        if not items:
            return None

        canonical = ";".join(sorted(set(items))) + ";"
        return hashlib.md5(canonical.encode("utf-8")).hexdigest().upper()

    def _derive_key(self, properties: dict[str, str]) -> bytes:
        """Derive the encryption key from file properties.

        :param properties: The properties of the HPS file.
        :return: The derived encryption key.
        """
        base_key = self._key_provider.get_key(properties)
        encryption_key_id = properties.get(self.HPS_ATTR_ENCRYPTION_KEY_ID)
        package_hash = self._compute_package_lock_hash(properties)

        if not encryption_key_id:
            return package_hash.encode("iso-8859-1") if package_hash else base_key

        if encryption_key_id == self.ENCRYPTION_KEY_ID_3SHAPE_INTERNAL and package_hash:
            return (base_key.decode("iso-8859-1") + package_hash).encode("iso-8859-1")

        return base_key
