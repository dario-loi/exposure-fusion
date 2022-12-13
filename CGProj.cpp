/*
 * \file   CGProj.cpp
 * \brief  Main application for Computer graphics project.
 *
 * \author Dario Loi, Flavio Gezzi
 * \date   December 2022
 */

#include "hdr.h"
#include "vendor/CImg.h"
#include <cstdio>
#include <iostream>
#include <omp.h>

const constexpr size_t X = 256;
const constexpr size_t Y = 256;

int main()
{
    namespace imglib = cimg_library;

    imglib::CImg<float> img{256, 256, 1, 3};

    img.draw_text(X / 2, Y / 2, "Hello World!", "white", 1.0f, 14);
    img.display();

    double x = hdr::stdev_fold<double>(1.0f, 2.0f, 3.0f, 1000.f, 0.00002f);
    printf("%f", x);

#pragma omp parallel for
    for (int i = 0; i < 16; i++)
    {
        if (omp_get_thread_num() == 0)
        {
            printf("We have %d threads working in parallel.\n ", omp_get_num_threads());
        }

        printf("Hi, I am thread %d\n", omp_get_thread_num());
    }

    return 1;
}
