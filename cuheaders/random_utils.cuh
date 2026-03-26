#ifndef RANDOMUTILS_H
#define RANDOMUTILS_H

#include <stdlib.h>
#include "vec3.cuh"

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__device__ inline vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0*RANDVEC3;
    } while (p.squared_length() >= 1.0);
    return p;
}

__device__ inline vec3 random_in_unit_disk(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0*vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1,1,0);
    } while (dot(p,p) >= 1.0);
    return p;
}

#endif
