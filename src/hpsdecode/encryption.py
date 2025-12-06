"""Encryption utilities for decrypting HPS files."""

from __future__ import annotations

__all__ = [
    "BlowfishDecryptor",
    "EncryptionKeyProvider",
    "EnvironmentKeyProvider",
    "StaticKeyProvider",
]

import abc
import os
import struct

from Crypto.Cipher import Blowfish

from hpsdecode.exceptions import HPSEncryptionError


def swap_endianness(data: bytes) -> bytes:
    """Swap byte order of 32-bit words in each 8-byte block.

    :param data: The input data (length must be multiple of 8).
    :return: The data with swapped endianness.
    """
    result = bytearray(len(data))

    for i in range(0, len(data), 8):
        block = data[i : i + 8]
        if len(block) == 8:
            left, right = struct.unpack("<II", block)
            result[i : i + 8] = struct.pack(">II", left, right)
        else:
            result[i : i + len(block)] = block

    return bytes(result)


class EncryptionKeyProvider(abc.ABC):
    """Base class for encryption key providers.

    Implement this interface to provide custom key resolution logic,
    such as loading keys from a configuration file.
    """

    @abc.abstractmethod
    def get_key(self, properties: dict[str, str]) -> bytes:
        """Retrieve the encryption key for the given file properties.

        :param properties: The properties of the HPS file.
        :return: The encryption key as bytes.
        :raises HPSEncryptionError: If the key cannot be retrieved.
        """
        pass


class EnvironmentKeyProvider(EncryptionKeyProvider):
    """Key provider that reads the encryption key from an environment variable.

    The key can be provided as either:

    - Comma-separated list of byte values (e.g., ``"10,20,30,40,50,..."``).
    - A raw string that will be encoded using ISO-8859-1.

    .. code-block:: python

        # Set the environment variable
        # export HPS_ENCRYPTION_KEY="10,20,30,40,50,..."

        provider = EnvironmentKeyProvider()
        key = provider.get_key({})

    """

    DEFAULT_ENV_VAR: str = "HPS_ENCRYPTION_KEY"

    _env_var: str

    def __init__(self, env_var: str = DEFAULT_ENV_VAR) -> None:
        """Initialize the environment key provider.

        :param env_var: The name of the environment variable to read.
        """
        self._env_var = env_var

    def get_key(self, properties: dict[str, str]) -> bytes:
        """Retrieve the encryption key from the environment.

        :param properties: The properties of the HPS file.
        :return: The encryption key as bytes.
        :raises HPSEncryptionError: If the environment variable is not set.
        """
        value = os.environ.get(self._env_var)
        if value is None:
            raise HPSEncryptionError(
                f"Encryption key not found. Set the {self._env_var} environment "
                f"variable or provide a key directly to the parser."
            )

        return self._parse_key(value)

    def _parse_key(self, value: str) -> bytes:
        """Parse the key value from string format.

        :param value: The key as comma-separated integers or raw string.
        :return: The key as bytes.
        """
        if "," in value:
            try:
                return bytes(int(b.strip()) for b in value.split(","))
            except ValueError:
                pass

        return value.encode("iso-8859-1")


class StaticKeyProvider(EncryptionKeyProvider):
    """Key provider that returns a fixed encryption key.

    Use this when you have a known key and want to pass it directly.

    code-block:: python

        key = bytes([10, 20, 30, 40, 50, ...])
        provider = StaticKeyProvider(key)

    """

    _key: bytes

    def __init__(self, key: bytes) -> None:
        """Initialize with a static key.

        :param key: The encryption key.
        """
        self._key = key

    def get_key(self, properties: dict[str, str]) -> bytes:
        """Return the static key.

        :param properties: The properties of the HPS file.
        :return: The static encryption key.
        """
        return self._key


class BlowfishDecryptor:
    """Blowfish ECB decryptor with endianness correction."""

    _cipher: Blowfish

    def __init__(self, key: bytes) -> None:
        """Initialize the decryptor with the given key.

        :param key: The encryption key, which must be between 5 and 56 bytes.
        """
        self._cipher = Blowfish.new(key, Blowfish.MODE_ECB)

    def decrypt(self, data: bytes, original_size: int | None = None) -> bytes:
        """Decrypt data using Blowfish ECB with endianness correction.

        :param data: The encrypted data. Will be padded to 8-byte boundary if needed.
        :param original_size: Original size before padding. If provided,
            output is truncated to this size to remove padding.
        :return: The decrypted data.
        """
        if not data:
            return data

        if len(data) % 8 != 0:
            padding = 8 - (len(data) % 8)
            data = data + bytes(padding)

        swapped = swap_endianness(data)
        decrypted = self._cipher.decrypt(swapped)
        result = swap_endianness(decrypted)

        if original_size is not None and original_size < len(result):
            result = result[:original_size]

        return result
