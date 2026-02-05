import argparse
import pathlib

from hpsdecode import load_hps


def format_bytes(size: int) -> str:
    """Format a byte size into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} TB"


def main() -> None:
    """Inspect an HPS scan file and print its metadata."""
    parser = argparse.ArgumentParser(description="Inspect HPS scan files.")
    parser.add_argument("input", type=pathlib.Path, help="Path to the HPS scan file to inspect.")

    args = parser.parse_args()
    if not args.input.exists():
        print(f"Error: File '{args.input}' does not exist.")
        return

    print(f"=== Inspecting: {args.input.name} ===")
    print(f"File Size: {format_bytes(args.input.stat().st_size)}")

    try:
        packed, mesh = load_hps(args.input)
    except Exception as e:
        print(f"Error loading HPS file: {e}")
        return

    print("\n[Metadata]")
    print(f"  Schema: {packed.schema}")
    print(f"  Encrypted: {'Yes' if packed.is_encrypted else 'No'}")
    print(f"  Expected Vertices: {packed.num_vertices}")
    print(f"  Expected Faces: {packed.num_faces}")

    print("\n[Decoded Mesh]")
    print(f"  Actual Vertices: {mesh.num_vertices}")
    print(f"  Actual Faces: {mesh.num_faces}")

    v_match = "✓" if packed.num_vertices == mesh.num_vertices else "✗"
    f_match = "✓" if packed.num_faces == mesh.num_faces else "✗"
    print(f"  Integrity: Vertices {v_match}, Faces {f_match}")

    print("\n[Attributes]")
    print(f"  Vertex Colors: {'Yes' if mesh.vertex_colors.size > 0 else 'No'}")
    print(f"  Face Colors: {'Yes' if mesh.face_colors.size > 0 else 'No'}")
    print(f"  Texture Coordinates: {'Yes' if mesh.uv.size > 0 else 'No'}")

    if mesh.num_vertices > 0:
        bounds_min = mesh.vertices.min(axis=0)
        bounds_max = mesh.vertices.max(axis=0)
        dimensions = bounds_max - bounds_min

        print("\n[Dimensions]")
        print(f"  Bounds: {dimensions[0]:.3f} x {dimensions[1]:.3f} x {dimensions[2]:.3f}")


if __name__ == "__main__":
    main()
