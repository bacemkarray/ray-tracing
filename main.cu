#include <iostream>
#include <fstream>
#include <limits>
#include <float.h>
#include <curand_kernel.h>

#include "cuheaders/vec3.cuh"
#include "cuheaders/ray.cuh"
#include "cuheaders/camera.cuh"
#include "cuheaders/hitable.cuh"
#include "cuheaders/hitable_list.cuh"
#include "cuheaders/sphere.cuh"


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, const char *const func, char const *const file, const int line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4.0f*a*c;
    return (discriminant > 0.0f);
}

__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
   ray cur_ray = r;
   vec3 cur_attenuation = vec3(1.0,1.0,1.0);

   for(int i = 0; i < 50; i++) {
      hit_record rec;
      if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }            
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
      }   
      return vec3(0.0, 0.0, 0.0); // exceeded recursion
}


__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   if((i >= max_x) || (j >= max_y)) return;
   int pixel_index = j*max_x + i;
   //Each thread gets same seed, a different sequence number, no offset
   curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}


__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;

    
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u,v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5,0.5,0.5)));
        int i = 1;
        for (int a=-11; a<11; a++) {
            for (int b=-11; b<11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
        
                if ((center-vec3(4,0.2,0)).length() > 0.9) {
                        if (choose_mat < 0.8f) { // diffuse
                            d_list[i++] = new sphere(center, 0.2, 
                                new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                        }
                        else if (choose_mat < 0.95f) { // metal
                            d_list[i++] = new sphere(center, 0.2, 
                                new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                        }
                        else { //glass
                            d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                        }
                    }
                }
            }
        d_list[i++] = new sphere(vec3(0,1,0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4,1,0), 1.0, new lambertian(vec3(0.4,0.2,0.1)));
        d_list[i++] = new sphere(vec3(4,1,0), 1.0, new metal(vec3(0.7,0.6,0.5), 0.0));
        *d_world = new hitable_list(d_list, i);
        
        // write back to change the state. so that the next time rand_state is used it wont keep generating the same numbers.
        *rand_state = local_rand_state;
        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; 
        //(lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom, lookat, vec3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus);
    }
}


__global__ void free_world(hitable **d_world, camera **d_camera) {
    delete *d_world;
    delete *d_camera;
}
    


int main() {
    int nx = 1200;
    int ny = 608;
    int ns = 100;
    int n = 500;
    int tx = 16;
    int ty = 16;
    
    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(float);

    //allocate pointers to gpu
    vec3 *fb;
    hitable **d_list;
    hitable **d_world;
    camera **d_camera;
    curandState *d_rand_state;
    curandState *d_rand_state2;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
    checkCudaErrors(cudaMalloc((void **)&d_list, n*sizeof(hitable *)));
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *))); 
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();    
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

  
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::ofstream MyFile("render.ppm");
    MyFile << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        // vec3 vector(0, 0, 0);
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            MyFile << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    //free objeects
    free_world<<<1,1>>>(d_world, d_camera);
    checkCudaErrors(cudaDeviceSynchronize());
    //free pointers on gpu
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));

    // useful for compute-sanitizer --leak-check full
    cudaDeviceReset();

    return 0;

    MyFile.close();

}
 