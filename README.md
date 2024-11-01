<!--
 * @Author: Xu.WANG
 * @Date: 2021-10-05 21:02:47
 * @LastEditTime: 2023-08-12 16:28:02
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @Description: 
-->
# XProtoSphere: an eXtended multi-sized sphere packing algorithm driven by particle size distribution (CGI 2023)

*Xu Wang, Makoto Fujisawa, Masahiko Mikawa

![](./pics/teaser.png)

[[Project Page]](https://raymondmcguire.github.io/xprotosphere/) [[Springer Nature Sharedit]](https://rdcu.be/dgafz)

This project implements an algorithm for packing multi-sized particles inside an arbitrary geometric object. Since this method is an extension version of the ProtoSphere method, we call this algorithm XProtoSphere. For detailed information, please refer to our paper.

## Environment Requirements

- C++17 or higher
- [CUDA](https://developer.nvidia.com/cuda-downloads) (Tested with CUDA 12.3 and 12.6)
- [Cmake](https://cmake.org/download/) 3.26 (recommended, newer versions may cause compatibility issues)
- Visual Studio 2022
- Blender 4.0 (for visualization)

## Installation

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/RaymondMcGuire/XProtoSphere.git
```

If you forgot to use --recursive, you can initialize submodules manually:
```bash
git submodule update --init --recursive
```

2. Install required software:

- Install CUDA 
- Install CMake
- Install Visual Studio 2022

## How to Build and Run

### For Windows

1. Navigate to the scripts folder:
```bash
cd ./scripts
```

2. Run the following batch files in order:
```bash
build_vs2022_win64.bat
compile_vs2022_release.bat
run_example.bat
```

The results will be exported to the "./export" folder.

## Visualization

You can use the blender template in the "./blender_template" folder to visualize the packing results.

![Example](./pics/bunny.png)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
