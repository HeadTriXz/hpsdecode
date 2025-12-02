import argparse
import pathlib

import trimesh

from hpsdecode import load_hps


def main() -> None:
    """Visualize an HPS scan file using trimesh."""
    parser = argparse.ArgumentParser(description="Inspect HPS scan files.")
    parser.add_argument("input", type=pathlib.Path, help="Path to the HPS scan file to inspect.")

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

    t_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    if mesh.vertex_colors.size > 0:
        t_mesh.visual.vertex_colors = mesh.vertex_colors

    t_mesh.show(caption=args.input.name)


if __name__ == "__main__":
    main()
