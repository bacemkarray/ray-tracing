#ifndef RANDOMUTILS_H
#define RANDOMUTILS_H

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <random>
#include "vec3.h"

inline float random_float() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

inline vec3 random_in_unit_sphere() {
    vec3 p;
    do {
        p = 2.0*vec3(random_float(),random_float(),random_float()) - vec3(1,1,1);
    } while (p.squared_length() >= 1.0);
    return p;
}

#endif
