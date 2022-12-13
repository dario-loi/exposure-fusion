/*
 * \file   hdr.h
 * \brief  functions for HDR
 *
 * \authors Dario Loi, Flavio Gezzi
 * \date   December 2022
 */

#pragma once

#include "vendor/CImg.h"
#include <stdint.h>

namespace hdr
{

/**
 * @brief Value of the variance of the exposure
 * @see compute_wexp
 */
constexpr const float EXPOSITION_VARIANCE = 0.2f;
using namespace cimg_library;

/**
 * @brief Fold expression for multiplication
 *
 * @note non l'ho mai fatto, ma l'ho sempre sognato.
 *
 * @author Dario Loi
 * @date December 2022
 *
 * @tparam Args a variadic template parameter pack
 * @tparam T a type that supports multiplication
 * @param args a parameter pack, containing the values to be multiplied
 * @return constexpr T the result of the multiplication
 */
template <typename T, typename... Args> constexpr inline T mul_fold(Args &&...args)
{
    return (... * args);
}

/**
 * @brief Copile time power function
 * 
 * @tparam T type of the base
 * @tparam Y exponent
 * @param x base
 * @return constexpr T result of the power, $x^y$
 */
template <typename T, uint64_t Y> constexpr inline T c_pow(T x)
{
    if constexpr (Y == 0ui64)
    {
        return static_cast<T>(1);
    }
    else if (Y == 1ui64)
    {
        return x;
    }
    else
        return x * c_pow<T, Y - 1>(x);
}

/**
 * @brief Fold expression for mean
 * 
 * @tparam T type of the samples
 * @tparam Args variadic template parameter pack
 * @param args samples
 * @return constexpr T mean of the samples
 */
template <typename T, typename... Args> constexpr inline T mean_fold(Args &&...args)
{
    return ((... + args) / sizeof...(args));
}

/**
 * @brief Fold expression for variance
 * 
 * @note this is the implementation of the fold expression for variance, it is not supposed to be called directly.
 * 
 * @tparam T type of the samples
 * @tparam Args variadic template parameter pack
 * @param args samples
 * @param mean mean of the samples
 * @return constexpr T variance of the samples
 */
template <typename T, typename... Args> constexpr inline T stdev_fold_impl(Args &&...args, T mean)
{
    return ((static_cast<T>(1) / ((sizeof...(args)) - static_cast<T>(1))) *
            (static_cast<T>(0) + ... + c_pow<T, 2>(args - mean)));
}

/**
 * @brief Fold expression for variance
 * 
 * @tparam T type of the samples
 * @tparam Args variadic template parameter pack
 * @param args samples
 * @return constexpr T variance of the samples
 */
template <typename T, typename... Args> constexpr inline T stdev_fold(Args &&...args)
{
    return stdev_fold_impl<T, Args...>(std::forward<Args>(args)..., mean_fold<T>(std::forward<Args>(args)...));
}

/**
 * @brief Map LDR images to HDR
 *
 * @authors Dario Loi, Flavio Gezzi
 * @date December 2022
 *
 * Implements the Exposure Fusion algorithm described in Mertens et al. combines
 * the information of multiple Low Definition Range (LDR) images to reconstruct
 * a High Definition image (HDR).
 *
 * @see https://web.stanford.edu/class/cs231m/project-1/exposure-fusion.pdf
 *
 * @tparam T the integral type of the pixel
 * @param LDR_images a CImg<T> object containing the LDR images
 * @return CImg<T> a CImg<T> object containing the HDR image
 */
template <typename T> CImg<T> LDR_to_HDR(CImg<T> const &LDR_images);

/**
 * @brief Compute the weights for each LDR image
 *
 * Computes the normalized weight matrix for a stack of LDR images.
 *
 * @author Dario Loi
 * @date December 2022
 *
 * @tparam T the integral type of the pixel
 * @param LDR_images a CImg<T> object containing the LDR images
 * @return CImg<T> a CImg<T> object containing the weights for each LDR image
 */
template <typename T> CImg<T> compute_W(CImg<T> const &LDR_images)
{

    auto C = compute_contrast(LDR_images);
    auto S = compute_saturation(LDR_images);
    auto E = compute_wexp(LDR_images);

    auto W = C.get_mul(S).get_mul(E);

    auto &W_normalized = W; // alias

    // W also changes, but more legible
    W_normalized = W.cumulate(2).get_invert() * W;

    return W_normalized;
}

/**
 * @brief Compute the Contrast of the LDR images
 *
 * Computes a contrast metric by performing a Laplacian convolution
 * on the LDR images.
 *
 * @author Flavio Gezzi
 * @date December 2022
 *
 * @tparam T the integral type of the pixel
 * @param LDR_images a CImg<T> object containing the LDR images
 * @return CImg<T> a CImg<T> object containing the contrast metric of the LDR images
 */
template <typename T> CImg<T> compute_contrast(CImg<T> const &LDR_images);
/**
 * @brief Compute the saturation of the LDR images
 *
 * @author Flavio Gezzi
 * @date December 2022
 *
 * @tparam T the integral type of the pixel
 * @param LDR_images a CImg<T> object containing the LDR images
 * @return CImg<T> a CImg<T> object containing the saturation of the LDR images
 */
template <typename T> CImg<T> compute_saturation(CImg<T> const &LDR_images);

/**
 * @brief Compute the exposure of the LDR images
 *
 * @author Dario Loi
 * @date December 2022
 *
 * @tparam T the integral type of the pixel
 * @param LDR_images a CImg<T> object containing the LDR images
 * @return CImg<T> a CImg<T> object containing the exposure of the LDR images
 */
template <typename T> CImg<T> compute_wexp(CImg<T> const &LDR_images)
{
    auto E = CImg<T>{LDR_images, "x,y,z,1"};

// If omp is available we multi-thread the whole process.
#pragma omp parallel for collapse(3)
    cimg_forXYZ(E, x, y, z)
    {
        T buf[E.spectrum]{}; // should be of tiny size (three in most cases)
        cimg_forC(E, c)
        {
            auto const &i = E(x, y, z, c);
            buf[c] = exp(-pow(i - 0.5f, 2) / (2 * EXPOSITION_VARIANCE * EXPOSITION_VARIANCE));
        }

        E(x, y, z, 1) = mul_fold(buf[0], buf[1], buf[2]);
    }

    return E;
}

/**
 * @brief Fuse the LDR images
 *
 *
 *
 * @tparam T the integral type of the pixel
 * @param LDR_images a CImg<T> object containing the LDR images
 * @param weights a CImg<T> object containing the weights for each LDR image
 * @return CImg<T>
 */
template <typename T> CImg<T> fuse_LDR(CImg<T> const &LDR_images, CImg<T> const &weights);

/**
 * @brief Reconstruction of the HDR image from the Laplacian pyramid
 *
 * @tparam T the integral type of the pixel
 * @param fused_laplacian a CImg<T> object containing the Laplacian pyramid of the fused LDR images
 * @return CImg<T> a CImg<T> object containing the final HDR image
 */
template <typename T> CImg<T> reconstruct_pyramid(CImg<T> const &fused_laplacian);

} // namespace hdr
