<div align="center">

# hpsdecode

[![Build Status][badge-build]][link-build]
[![PyPI Version][badge-pypi]][link-pypi]
[![Python Version][badge-python]][link-repo]
[![License][badge-license]][link-repo]

</div>

A Python library for decoding HPS (HIMSA Packed Scan) and DCM files used by dental and audiological scanning software
such as [3Shape](https://www.3shape.com/) and [ShapeDesigner](https://www.mechatools.com/en/shapedesigner.html).

HPS is a compressed 3D mesh format commonly used in dental scanning applications and other HIMSA-compliant devices.

## Features

- Read mesh geometry (vertices and faces) from HPS files.
- Supports CA, CC, and CE schemas, including encrypted files.
- Extract mesh colors, texture coordinates (UVs) and texture images.
- Export the mesh to OBJ, PLY and STL.

<details>
<summary>Planned Features</summary>

- Extract splines and curves in the scan data.
- Command-line tools for conversion and visualization.

</details>


## Getting Started

### Installation

```sh
pip install hpsdecode
```

### Basic Usage

```python
from hpsdecode import load_hps

# Load from file path
packed, mesh = load_hps("scan.dcm")

# Access mesh data
print(f"Vertices: {mesh.num_vertices}")
print(f"Faces: {mesh.num_faces}")
print(f"Schema: {packed.schema}")

# Vertex positions as (N, 3) float32 array
vertices = mesh.vertices

# Face indices as (M, 3) int32 array  
faces = mesh.faces
```

## Exporting Meshes

After loading and decoding an HPS file, you can export the mesh to common 3D file formats for use in other software or workflows:

```python
mesh.export("output.obj")   # OBJ
mesh.export("output.ply")   # PLY
mesh.export("output.stl")   # STL
```

### Format Support

| Format  | Vertices & Faces | Vertex Colors | Textures | Binary/ASCII |
|---------|:----------------:|:-------------:|:--------:|:------------:|
| **OBJ** |        ✅         |      ✅*       |    ✅     |    ASCII     |
| **PLY** |        ✅         |       ✅       |   ✅**    | ASCII/Binary |
| **STL** |        ✅         |       ❌       |    ❌     | ASCII/Binary |

<sup>* OBJ vertex colors use a non-standard extension; not all software supports this.</sup>  
<sup>** PLY textures are baked into vertex colors. This may result in loss of quality.</sup>

## Compression Schemas

| Schema | Status        | Description                                                                        |
|--------|---------------|------------------------------------------------------------------------------------|
| **CA** | ✅ Supported   | Identical to CC; provided for backward compatibility.                              |
| **CB** | ❌ Not Planned | Lossy compression with optional color and texture data. No example data available. |
| **CC** | ✅ Supported   | Lossless compression with uncompressed vertices and compressed faces.              |
| **CE** | ✅ Supported   | Encrypted version of CC. Requires an encryption key.                               |

## Encrypted Files (CE Schema)

Some HPS files use the CE schema, which encrypts the mesh data. To decode these files, you must provide the encryption key.

> [!NOTE]
> This library does not provide encryption keys, nor will it provide instructions on how to obtain them.

### Providing the Encryption Key

**Option 1: Environment Variable (Recommended)**

Set the `HPS_ENCRYPTION_KEY` environment variable before loading files:

```sh
export HPS_ENCRYPTION_KEY="28,141,16,74,219,32,11,126,55,178,97,3,41,82,213,222"
```

```python
from hpsdecode import load_hps

# Automatically uses the key from the environment variable
packed, mesh = load_hps("encrypted.hps")
```

**Option 2: Direct Key**

Provide the key directly as a bytes object:

```python
from hpsdecode import load_hps

key = bytes([28, 141, 16, 74, 219, 32, 11, 126, 55, 178, 97, 3, 41, 82, 213, 222])

packed, mesh = load_hps("encrypted.hps", encryption_key=key)
```

**Option 3: Custom Key Provider**

Implement your own key provider for advanced use cases (e.g., loading from a configuration file):

```python
from hpsdecode import load_hps
from hpsdecode.encryption import EncryptionKeyProvider

class MyKeyProvider(EncryptionKeyProvider):
    def get_key(self, properties):
        return load_key_from_config()


key_provider = MyKeyProvider()
packed, mesh = load_hps("encrypted.hps", encryption_key=key_provider)
```

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

  > Run `pip install trimesh[recommend]` for viewing support.


## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or pull request if you would like to assist.

If you'd like to support development, consider donating to [my Ko-fi page][link-kofi]. Every contribution is highly appreciated!

[![ko-fi][badge-kofi]][link-kofi]


## License

This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.


<!-- Image References -->
[badge-build]:https://img.shields.io/github/actions/workflow/status/HeadTriXz/hpsdecode/code-quality.yml?style=for-the-badge
[badge-kofi]:https://ko-fi.com/img/githubbutton_sm.svg
[badge-license]:https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge
[badge-pypi]:https://img.shields.io/pypi/v/hpsdecode?style=for-the-badge
[badge-python]:https://img.shields.io/pypi/pyversions/hpsdecode?style=for-the-badge

<!-- Links -->
[link-build]:https://github.com/HeadTriXz/hpsdecode/actions/workflows/code-quality.yml
[link-kofi]:https://ko-fi.com/headtrixz
[link-license]:https://github.com/HeadTriXz/hpsdecode/blob/main/LICENSE
[link-pypi]:https://pypi.org/project/hpsdecode/
[link-repo]:https://github.com/HeadTriXz/hpsdecode
