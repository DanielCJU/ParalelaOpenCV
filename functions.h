#ifndef FUNCTIONS_H_INCLUDED
#define FUNCTIONS_H_INCLUDED

#include <iostream>
#include <cstdlib>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <mpi.h>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>

float linear_extrapolation(float, float, float);
float bi_linear_extrapolation(float, float, float, float, float, float);
void Generar_mascara(float[5][5]);
void obtener_fragmento(Mat, Mat, int, int, int, int);
void join_gaussian_blur(Mat, Mat, int, int);
void join_luminosity_scale(Mat, Mat, int, int);
void enviar(Mat, int);
void recibir(&Mat, int);
void Gaussian_blur(Mat, Mat, int, int);
void Average(Mat, Mat, int, int);
cv::Mat bi_lineal_scale(Mat , float);

#endif // FUNCTIONS_H_INCLUDED
