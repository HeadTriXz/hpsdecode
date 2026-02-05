import argparse
import io
import pathlib

import numpy as np
import trimesh
import trimesh.visual
from PIL import Image

from hpsdecode import load_hps


def convert_bgr_to_rgb(image_data: bytes) -> Image.Image:
    """Convert BGR image data to RGB PIL Image.

    :param image_data: The raw image bytes (JPEG or PNG in BGR format).
    :return: A PIL Image in RGB format.
    """
    img = Image.open(io.BytesIO(image_data))
    if img.mode != "RGB":
        img = img.convert("RGB")

    r, g, b = img.split()
    return Image.merge("RGB", (b, g, r))


def create_texture_visual(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    texture_data: bytes,
) -> trimesh.Trimesh:
    """Create a textured mesh from per-corner UV coordinates.

    :param vertices: The original vertex positions (N, 3).
    :param faces: The original face indices (M, 3).
    :param uv_coords: The per-corner UV coordinates (M * 3, 2).
    :param texture_data: The raw texture image data (BGR format).
    :return: A new trimesh with duplicated vertices at UV seams.
    """
    texture_image = convert_bgr_to_rgb(texture_data)
    texture_image = texture_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    num_faces = faces.shape[0]
    uv_coords = uv_coords.reshape(num_faces, 3, 2)

    vertex_uv_map: dict[tuple[int, tuple[float, float]], int] = {}
    new_vertices = []
    new_uvs = []
    new_faces = []

    for face_idx, face in enumerate(faces):
        new_face = []
        for corner_idx in range(3):
            vertex_idx = face[corner_idx]
            uv = tuple(uv_coords[face_idx, corner_idx])

            key = (vertex_idx, uv)
            if key not in vertex_uv_map:
                unique_idx = len(new_vertices)
                vertex_uv_map[key] = unique_idx
                new_vertices.append(vertices[vertex_idx])
                new_uvs.append(uv)

            new_face.append(vertex_uv_map[key])

        new_faces.append(new_face)

    new_vertices = np.array(new_vertices, dtype=np.float32)
    new_uvs = np.array(new_uvs, dtype=np.float32)
    new_faces = np.array(new_faces, dtype=np.int32)

    material = trimesh.visual.material.SimpleMaterial(image=texture_image)
    visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=material)

    return trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        visual=visual,
        process=False,
    )


def main() -> None:
    """Visualize an HPS scan file using trimesh."""
    parser = argparse.ArgumentParser(description="Visualize HPS scan files.")
    parser.add_argument("input", type=pathlib.Path, help="Path to the HPS scan file to visualize.")
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Disable texture rendering even if available.",
    )

    args = parser.parse_args()
    if not args.input.exists():
        print(f"Error: File '{args.input}' does not exist.")
        return

    print(f"Loading {args.input.name}...")
    try:
        packed, mesh = load_hps(args.input)
    except Exception as e:
        print(f"Error loading HPS file: {e}")
        return

    has_texture = mesh.has_textures and mesh.has_texture_coords and not args.no_texture
    has_vertex_colors = mesh.has_vertex_colors
    has_face_colors = mesh.has_face_colors

    if has_texture:
        print(f"Applying texture (using first of {len(mesh.texture_images)} images)...")
        try:
            t_mesh = create_texture_visual(mesh.vertices, mesh.faces, mesh.uv, mesh.texture_images[0])
        except Exception as e:
            print(f"Failed to apply texture: {e}")
            print("Falling back to color rendering...")
            has_texture = False

    if not has_texture:
        t_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=mesh.vertex_colors if has_vertex_colors else None,
            face_colors=mesh.face_colors if has_face_colors and not has_vertex_colors else None,
            process=False,
        )

    t_mesh.show(caption=args.input.name, smooth=False)


if __name__ == "__main__":
    main()
