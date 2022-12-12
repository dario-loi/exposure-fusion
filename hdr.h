/*****************************************************************//**
 * \file   hdr.h
 * \brief  functions for HDR
 * 
 * \author Dario Loi
 * \date   December 2022
 *********************************************************************/

#pragma once

#include "vendor/CImg.h"
using namespace cimg_library;

/**
 * @brief Map LDR images to HDR
 * 
 * @tparam T 
 * @param LDR_images 
 * @return CImg<T> 
 */
template<typename T>
CImg<T> LDR_to_HDR(CImg<T> const& LDR_images);

/**
 * @brief Compute the weights for each LDR image
 * @author Dario Loi
 * @tparam T the integral type of the pixel
 * @param LDR_images a CImg<T> object containing the LDR images
 * @return CImg<T> a CImg<T> object containing the weights for each LDR image
 */
template<typename T> 
CImg<T> compute_W(CImg<T> const& LDR_images);

/**
 * @brief Compute the Laplacian pyramid of the LDR images
 * @author Flavio Gezzi
 * @tparam T the integral type of the pixel
 * @param LDR_images a CImg<T> object containing the LDR images
 * @return CImg<T> a CImg<T> object containing the Laplacian pyramid of the LDR images
 */
template<typename T>
CImg<T> compute_contrast(CImg<T> const& LDR_images);

/**
 * @brief Compute the saturation of the LDR images
 * @author Flavio Gezzi
 * @tparam T the integral type of the pixel
 * @param LDR_images a CImg<T> object containing the LDR images
 * @return CImg<T> a CImg<T> object containing the saturation of the LDR images
 */
template<typename T>
CImg<T> compute_saturation(CImg<T> const& LDR_images);

/**
 * @brief Compute the exposure of the LDR images
 * @author Dario Loi
 * @tparam T the integral type of the pixel
 * @param LDR_images 
 * @return CImg<T> a CImg<T> object containing the exposure of the LDR images
 */
template<typename T>
CImg<T> compute_wexp(CImg<T> const& LDR_images);


/**
 * @brief Fuse the LDR images
 * 
 * @tparam T the integral type of the pixel
 * @param LDR_images 
 * @param weights 
 * @return CImg<T> 
 */
template<typename T>
CImg<T> fuse_LDR(CImg<T> const& LDR_images, CImg<T> const& weights);

/**
 * @brief Reconstruction of the HDR image from the Laplacian pyramid
 * 
 * @tparam T the integral type of the pixel
 * @param fused_laplacian 
 * @return CImg<T> 
 */
template<typename T>
CImg<T> reconstruct_pyramid(CImg<T> const& fused_laplacian);

