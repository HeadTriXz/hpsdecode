"""Command-line interface for hpsdecode."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from hpsdecode import load_hps
from hpsdecode.exceptions import HPSEncryptionError, HPSParseError, HPSSchemaError
from hpsdecode.export import ExportFormat
from hpsdecode.export.obj import MaterialConfig


def load_encryption_key(key_arg: str | None) -> bytes | None:
    """Load and parse an encryption key from various formats.

    :param key_arg: The key string from command-line argument, or None to check environment.
    :return: The parsed encryption key as bytes, or None if no key is available.
    :raises ValueError: If the key is invalid.
    """
    if key_arg is None:
        env_key = os.environ.get("HPS_ENCRYPTION_KEY")
        if env_key is None:
            return None

        key_arg = env_key

    key_arg = key_arg.strip()
    if "," in key_arg:
        try:
            return bytes(int(b.strip()) for b in key_arg.split(","))
        except ValueError as e:
            raise ValueError(f"Invalid comma-separated key format: {e}") from None

    if key_arg.startswith(("0x", "0X")):
        key_arg = key_arg[2:]

    try:
        return bytes.fromhex(key_arg)
    except ValueError:
        pass

    return key_arg.encode("iso-8859-1")


def format_bytes(size: int) -> str:
    """Format a byte size into a human-readable string.

    :param size: The size in bytes.
    :return: A formatted string with appropriate units (B, KB, MB, GB, TB).
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} TB"


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    :return: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="hpsdecode",
        description="Decode and export HIMSA Packed Standard (HPS) files",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export",
        help="Export an HPS file to a common 3D format",
        description="Export an HPS file to STL, OBJ, or PLY format",
    )

    export_parser.add_argument(
        "input",
        type=Path,
        help="path to the input HPS file",
    )

    export_parser.add_argument(
        "output",
        type=Path,
        help="path where the exported file will be written",
    )

    export_parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["stl", "obj", "ply"],
        help="output file format (automatically detected from file extension if omitted)",
    )

    export_parser.add_argument(
        "--ascii",
        action="store_true",
        help="export in ASCII text format instead of binary (STL and PLY only)",
    )

    export_parser.add_argument(
        "--no-colors",
        action="store_true",
        help="exclude per-vertex color data from the exported file (OBJ and PLY only)",
    )

    export_parser.add_argument(
        "--no-textures",
        action="store_true",
        help="exclude texture coordinates and texture images from the exported file (OBJ and PLY only)",
    )

    export_parser.add_argument(
        "-k",
        "--key",
        type=str,
        metavar="KEY",
        help=(
            "encryption key for encrypted HPS files. "
            "Accepts raw string, hex string (0x1c8d10...) or comma-separated byte values (28,141,16,...). "
            "If not provided, the key will be read from the HPS_ENCRYPTION_KEY environment variable."
        ),
    )

    material_group = export_parser.add_argument_group(
        "material options",
        "Material properties for OBJ export (MTL file generation)",
    )

    material_group.add_argument(
        "--ambient",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="ambient reflectance color as RGB values (0.0-1.0)",
    )

    material_group.add_argument(
        "--diffuse",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="diffuse reflectance color as RGB values (0.0-1.0)",
    )

    material_group.add_argument(
        "--specular",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="specular reflectance color as RGB values (0.0-1.0)",
    )

    material_group.add_argument(
        "--shininess",
        type=float,
        metavar="NS",
        help="specular exponent controlling highlight sharpness (0.0-1000.0, where higher values create tighter highlights)",
    )

    material_group.add_argument(
        "--optical-density",
        type=float,
        metavar="NI",
        help="optical density for transparent materials (0.001-10.0, where 1.0 is no refraction)",
    )

    material_group.add_argument(
        "--dissolve",
        type=float,
        metavar="D",
        help="opacity factor (0.0-1.0, where 1.0 is fully opaque and 0.0 is fully transparent)",
    )

    return parser


def parse_material_config(args: argparse.Namespace) -> MaterialConfig:
    """Parse material configuration from command-line arguments.

    :param args: The parsed command-line arguments.
    :return: A MaterialConfig object with the specified properties.
    """
    material = MaterialConfig()

    def validate_float(value: float, name: str, min_value: float, max_value: float) -> float:
        if not min_value <= value <= max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")

        return value

    def validate_rgb(values: list[float], name: str) -> tuple[float, float, float]:
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError(f"{name} values must be between 0.0 and 1.0")

        return tuple(values)

    if args.ambient is not None:
        material.ambient = validate_rgb(args.ambient, "Ambient")

    if args.diffuse is not None:
        material.diffuse = validate_rgb(args.diffuse, "Diffuse")

    if args.specular is not None:
        material.specular = validate_rgb(args.specular, "Specular")

    if args.shininess is not None:
        material.shininess = validate_float(args.shininess, "Shininess", 0.0, 1000.0)

    if args.optical_density is not None:
        material.optical_density = validate_float(args.optical_density, "Optical Density", 0.001, 10.0)

    if args.dissolve is not None:
        material.dissolve = validate_float(args.dissolve, "Dissolve", 0.0, 1.0)

    return material


def export_command(args: argparse.Namespace) -> int:
    """Execute the export command.

    :param args: The parsed command-line arguments.
    :return: Exit code (0 for success, non-zero for error).
    """
    try:
        encryption_key = load_encryption_key(args.key)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        _, mesh = load_hps(args.input, encryption_key)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    except HPSSchemaError as e:
        print(f"Error: Unsupported HPS schema: {e}", file=sys.stderr)
        return 1
    except HPSParseError as e:
        print(f"Error: Failed to parse HPS file: {e}", file=sys.stderr)
        return 1
    except HPSEncryptionError as e:
        print(f"Error: Decryption failed: {e}", file=sys.stderr)
        return 1

    export_format = ExportFormat(args.format) if args.format else None
    material = parse_material_config(args)

    try:
        mesh.export(
            args.output,
            export_format=export_format,
            binary=not args.ascii,
            include_colors=not args.no_colors,
            include_textures=not args.no_textures,
            material=material,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"Error: Failed to write output file: {e}", file=sys.stderr)
        return 1

    output_size = format_bytes(args.output.stat().st_size)
    print(f"✓ Successfully exported to '{args.output}' ({output_size})")

    return 0


def main() -> int:
    """Main entry point for the CLI.

    :return: The exit code (0 for success, non-zero for error).
    """
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "export":
        return export_command(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
