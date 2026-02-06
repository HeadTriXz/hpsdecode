"""Base interface for mesh exporters."""

from __future__ import annotations

__all__ = ["BaseExporter"]

import abc
import typing as t

if t.TYPE_CHECKING:
    import os

    from hpsdecode.mesh import HPSMesh


class BaseExporter(abc.ABC):
    """Abstract base class for mesh exporters."""

    @abc.abstractmethod
    def export(self, mesh: HPSMesh, output_path: str | os.PathLike[str]) -> None:
        """Export a mesh to a file.

        :param mesh: The mesh to export.
        :param output_path: The output file path.
        """
        raise NotImplementedError
