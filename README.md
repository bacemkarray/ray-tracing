
# Ray Tracer

(<img width="1200" height="608" alt="render" src="https://github.com/user-attachments/assets/95cb9221-a115-4e59-b252-7d6301a24990" />)

A GPU-accelerated ray tracer built with C++ and CUDA. This project includes both a traditional CPU-bound ray tracer and a parallelized implementation of it, designed to drastically reduce render times by leveraging the massive parallelism of GPUs.

## Implementation Details & Performance

To optimize the rendering process, I focused heavily on efficient thread and block allocation:

* **Grid & Block Dimensions:** The image is divided into chunks. I used a block dimension of `<insert your block dim, e.g., 8x8>` threads. I chose this because `<explain why, e.g., it perfectly aligns with warp sizes and maximizes occupancy>`.
* **Memory Management:** `<Mention if you used unified memory (cudaMallocManaged), constant memory, or any specific memory optimizations>`.
* **Performance Gains:** `<If you have a rough idea of how much faster this is than a standard C++ CPU render, mention it here! e.g., "Reduced a 10-minute CPU render down to 5 seconds">`.

## Getting Started

### Prerequisites
* NVIDIA GPU with CUDA support
* CUDA Toolkit (`nvcc`)
* C++ compiler

### Building the Project
This project uses a standard `Makefile` for easy compilation. 

```bash
# Clone the repository
git clone [https://github.com/bacemkarray/ray-tracing.git](https://github.com/bacemkarray/ray-tracing.git)
cd ray-tracing

# Build the executable
make

# Run the ray tracer
./main.exe > image.ppm
