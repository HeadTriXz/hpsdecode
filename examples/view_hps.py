import argparse
import io
import pathlib

import numpy as np
import numpy.typing as npt
import trimesh
import trimesh.visual
from PIL import Image

from hpsdecode import load_hps
from hpsdecode.mesh import Spline


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


def create_cylinder_mesh(
    start_point: npt.NDArray[np.floating],
    end_point: npt.NDArray[np.floating],
    radius: float,
    slices: int = 32,
) -> trimesh.Trimesh:
    """Create a cylinder mesh between two points.

    :param start_point: The starting point of the cylinder.
    :param end_point: The ending point of the cylinder.
    :param radius: The radius of the cylinder.
    :param slices: The number of slices around the cylinder circumference.
    :return: A Trimesh representing the cylinder.
    """
    direction = end_point - start_point
    height = np.linalg.norm(direction)
    if height < 1e-6:
        return trimesh.Trimesh()

    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=slices)

    z_axis = np.array([0.0, 0.0, 1.0])
    direction_normalized = direction / height

    rotation_matrix = trimesh.geometry.align_vectors(z_axis, direction_normalized)
    cylinder.apply_transform(rotation_matrix)

    center = (start_point + end_point) / 2.0
    cylinder.apply_translation(center)

    return cylinder


def create_spline_mesh(spline: Spline, slices: int = 32) -> trimesh.Trimesh:
    """Create a mesh representation of a spline as connected cylinders.

    :param spline: The spline to convert to a mesh.
    :param slices: The number of slices around each cylinder circumference.
    :return: A trimesh representing the spline.
    """
    if spline.num_control_points < 2:
        return trimesh.Trimesh()

    meshes = []
    control_points = spline.control_points

    for i in range(len(control_points) - 1):
        cylinder = create_cylinder_mesh(control_points[i], control_points[i + 1], spline.radius, slices)
        if cylinder.vertices.size > 0:
            meshes.append(cylinder)

    if spline.is_cyclic and len(control_points) >= 2:
        cylinder = create_cylinder_mesh(control_points[-1], control_points[0], spline.radius, slices)
        if cylinder.vertices.size > 0:
            meshes.append(cylinder)

    if not meshes:
        return trimesh.Trimesh()

    r = (spline.color >> 16) & 0xFF
    g = (spline.color >> 8) & 0xFF
    b = spline.color & 0xFF

    combined = trimesh.util.concatenate(meshes)
    combined.visual.vertex_colors = np.array([r, g, b, 255], dtype=np.uint8)

    return combined


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
    parser.add_argument(
        "--show-splines",
        action="store_true",
        help="Show splines associated with the scan, if available.",
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

    scene = trimesh.Scene()
    scene.add_geometry(t_mesh, node_name="mesh")

    if args.show_splines and mesh.has_splines:
        print(f"Creating spline visualization ({len(mesh.splines)} splines)...")

        for idx, spline in enumerate(mesh.splines):
            spline_mesh = create_spline_mesh(spline)
            if spline_mesh.vertices.size > 0:
                scene.add_geometry(spline_mesh, node_name=f"spline_{idx}")

    scene.show(caption=args.input.name, smooth=False)


if __name__ == "__main__":
    main()
