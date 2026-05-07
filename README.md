# Ray Tracer

<img width="1200" height="608" alt="render" src="https://github.com/user-attachments/assets/95cb9221-a115-4e59-b252-7d6301a24990" />

A small path tracer implemented in C++ with both CPU and CUDA renderers. Includes diffuse, metal, and dielectric materials; recursive ray scattering; antialiasing; gamma correction; and a depth-of-field camera.

The CUDA version parallelizes rendering across pixels so each GPU thread traces and samples one pixel independently. This makes it a good project for comparing a straightforward CPU renderer with a GPU implementation of the same ray-tracing logic.

## Features

- CPU renderer in `main.cpp`
- CUDA renderer in `main.cu`
- Randomized scene generation with hundreds of spheres
- Lambertian, metal, and dielectric materials
- Camera aperture and focus distance for depth of field
- Per-pixel random sampling with CURAND in the CUDA path
- PPM image output
- Makefile targets for building and profiling

## Implementation Details

The CPU and CUDA implementations share the same basic rendering model: rays are generated from a camera, tested against a list of hittable objects, scattered through materials up to a maximum depth, and accumulated into a final pixel color.

The CUDA renderer uses:

- `16x16` thread blocks
- One thread per pixel
- A `1200x608` output resolution
- `100` samples per pixel
- Up to `50` ray bounces per sample
- `cudaMallocManaged` for the framebuffer
- Device allocations for the world, camera, object list, and CURAND states

The scene and camera are created on the GPU, then the render kernel fills the framebuffer in parallel. Each pixel gets its own CURAND state so antialiasing and material scattering can use independent random samples.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit with `nvcc`, `nsys`, and `ncu`
- C++ compiler, configured as `g++` in the Makefile
- `make`

The Makefile assumes CUDA is installed at `/usr/local/cuda` unless `CUDA_PATH` is set.

## Building

Build the CPU renderer:

```bash
make cpuart
```

Build the CUDA renderer:

```bash
make cudart
```

On Windows, the executables are built as `cpuart.exe` and `cudart.exe`. On Linux/macOS-style environments, they are built as `cpuart` and `cudart`.

## Running

Run the CPU renderer:

```bash
./cpuart
```

Run the CUDA renderer:

```bash
./cudart
```

Both renderers write a PPM image to:

```text
render.ppm
```

The CUDA renderer also prints the render time to standard error.

## Profiling

Build and profile the CUDA renderer with NVIDIA Nsight Systems:

```bash
make profile_basic
```

Collect a compute profile with NVIDIA Nsight Compute:

```bash
make profile_compute
```

Collect selected occupancy and instruction metrics:

```bash
make profile_metrics
```

Profiling reports are written to the `reports/` directory. You can set a custom report tag with:

```bash
make profile_basic TAG=my_run
```

## Cleaning

Remove generated executables and output files:

```bash
make clean
```