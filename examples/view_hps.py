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


def create_texture_visual(uv_coords: np.ndarray, texture_data: bytes) -> trimesh.visual.TextureVisuals:
    """Create a TextureVisuals object from mesh and texture data.

    :param uv_coords: The UV coordinates (N, 2).
    :param texture_data: The raw texture image data (BGR format).
    :return: A TextureVisuals object.
    """
    texture_image = convert_bgr_to_rgb(texture_data)
    texture_image = texture_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    material = trimesh.visual.material.SimpleMaterial(image=texture_image)

    return trimesh.visual.TextureVisuals(uv=uv_coords, material=material)


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

    visual = None
    has_texture = mesh.has_textures and mesh.has_texture_coords and not args.no_texture
    has_vertex_colors = mesh.has_vertex_colors
    has_face_colors = mesh.has_face_colors

    if has_texture:
        print(f"Applying texture (using first of {len(mesh.texture_images)} images)...")
        try:
            visual = create_texture_visual(mesh.uv, mesh.texture_images[0])
        except Exception as e:
            print(f"Failed to apply texture: {e}")
            print("Falling back to color rendering...")
            has_texture = False

    t_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=mesh.vertex_colors if has_vertex_colors and not has_texture else None,
        face_colors=mesh.face_colors if has_face_colors and not has_texture and not has_vertex_colors else None,
        visual=visual,
        process=False,
    )

    t_mesh.show(caption=args.input.name, smooth=False)


if __name__ == "__main__":
    main()
