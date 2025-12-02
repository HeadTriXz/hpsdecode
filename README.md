<div align="center">

# hpsdecode

[![PyPI Version][badge-pypi]][link-pypi]
[![Python Version][badge-python]][link-repo]
[![License][badge-license]][link-repo]

</div>

A Python library for decoding HPS (HIMSA Packed Scan) files as specified in
the [Packed Scan Standard Format 501](https://himsafiles.com/DataStandards/DataStandards/Scan/Packed%20Scan%20Standard%20format%20501.pdf).

HPS is a compressed 3D mesh format commonly used in dental and audiological scanning applications, including 3Shape
dental scanners and other HIMSA-compliant devices.


## Getting Started

### Requirements

- Python 3.11+
- NumPy

### Installation

```sh
pip install hpsdecode
```

### Basic Usage

```python
from hpsdecode import load_hps

# Load from file path
packed_scan, mesh = load_hps("scan.hps")

# Access mesh data
print(f"Vertices: {mesh.num_vertices}")
print(f"Faces: {mesh.num_faces}")
print(f"Schema: {packed_scan.schema}")

# Vertex positions as (N, 3) float32 array
vertices = mesh.vertices

# Face indices as (M, 3) int32 array  
faces = mesh.faces
```


## Compression Schemas

| Schema | Status      | Description                                                                                       |
|--------|-------------|---------------------------------------------------------------------------------------------------|
| **CA** | ✅ Supported | Identical to CC; provided for backward compatibility.                                             |
| **CB** | 🚧 Planned  | Supports compression of both vertices and faces, with optional color and texture data.            |
| **CC** | ✅ Supported | Stores uncompressed vertices and compressed faces; does not support color or texture information. |
| **CE** | 🚧 Planned  | Reserved for future use; details of this schema are currently unknown.                            |

The CC compression schema is similar to CB but is simpler and fully lossless. It stores each vertex as three 32-bit
floats (x, y, z) and uses a 4-bit instruction set for faces. CC does not compress as much as CB and cannot store texture
or special commands like color changes.


## File Format Overview

HPS files are XML documents containing base64-encoded binary mesh data:

```xml

<HPS version="1.1">
    <Packed_geometry>
        <Schema>CA</Schema>
        <Binary_data>
            <CA version="1.0">
                <Vertices base64_encoded_bytes="..." vertex_count="...">
                    <!-- Base64-encoded float32 vertex positions -->
                </Vertices>
                <Facets base64_encoded_bytes="..." facet_count="...">
                    <!-- Base64-encoded face commands -->
                </Facets>
            </CA>
        </Binary_data>
    </Packed_geometry>
</HPS>
```

## Example Scripts

- **Inspect an HPS file:**  
  Print metadata, stats, and mesh extents to the console:

  ```bash
  python examples/inspect_hps.py path/to/file.hps
  ```

- **View HPS file in 3D:**  
  Visualize the mesh in an interactive viewer (requires `trimesh`):

  ```bash
  python examples/view_hps.py path/to/file.hps
  ```

  > [!NOTE]
  > Run `pip install trimesh[recommended]` for viewing support.


## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or pull request if you would like to assist.

If you'd like to support development, consider donating to [my Ko-fi page][link-kofi]. Every contribution is highly appreciated!

[![ko-fi][badge-kofi]][link-kofi]


## License

This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.


<!-- Image References -->
[badge-kofi]:https://ko-fi.com/img/githubbutton_sm.svg
[badge-license]:https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge
[badge-pypi]:https://img.shields.io/pypi/v/hpsdecode?style=for-the-badge
[badge-python]:https://img.shields.io/pypi/pyversions/hpsdecode?style=for-the-badge

<!-- Links -->
[link-kofi]:https://ko-fi.com/headtrixz
[link-license]:https://github.com/HeadTriXz/hpsdecode/blob/main/LICENSE
[link-pypi]:https://pypi.org/project/hpsdecode/
[link-repo]:https://github.com/HeadTriXz/hpsdecode
