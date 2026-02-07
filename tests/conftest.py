import io

import numpy as np
import pytest
from PIL import Image

from hpsdecode import HPSMesh


@pytest.fixture
def empty_mesh() -> HPSMesh:
    """An empty mesh with no vertices, faces, colors, or textures."""
    return HPSMesh(
        vertices=np.empty((0, 3), dtype=np.float32),
        faces=np.empty((0, 3), dtype=np.int32),
        vertex_colors=np.empty((0, 3), dtype=np.uint8),
        face_colors=np.empty((0, 3), dtype=np.uint8),
        uv=np.empty((0, 2), dtype=np.float32),
    )


@pytest.fixture
def simple_mesh() -> HPSMesh:
    """A simple mesh without colors or textures."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    return HPSMesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=np.array([]),
        face_colors=np.array([]),
        uv=np.array([]),
    )


@pytest.fixture
def colored_mesh() -> HPSMesh:
    """A mesh with vertex colors but no textures."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    vertex_colors = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )
    face_colors = np.array([[255, 128, 64]], dtype=np.uint8)

    return HPSMesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        face_colors=face_colors,
        uv=np.array([]),
    )


@pytest.fixture
def textured_mesh() -> HPSMesh:
    """A mesh with UV coordinates and a texture image."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uv = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    image_bytes = io.BytesIO()

    image = Image.new("RGB", (2, 2), color=(255, 128, 64))
    image.save(image_bytes, format="PNG")

    return HPSMesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=np.array([]),
        face_colors=np.array([]),
        uv=uv,
        texture_images=[image_bytes.getvalue()],
    )
