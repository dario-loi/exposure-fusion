/*****************************************************************//**
 * \file   CGProj.cpp
 * \brief  Main application for Computer graphics project.
 * 
 * \author Dario Loi
 * \date   December 2022
 *********************************************************************/

#include "vendor/CImg.h"

const constexpr size_t X = 256;
const constexpr size_t Y = 256;

int main()
{
	namespace imglib = cimg_library;

	imglib::CImg<float> img{ 256, 256, 1, 3 };

	img.draw_text(X / 2, Y / 2, "Hello World!", "white", 1.0f, 14);
	img.display();

	return 0;
}
