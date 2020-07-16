#include <iostream>
#include <cstdlib>/*
#include <opencv2/opencv_modules.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <mpi.h>*/
#include <string>
#include <vector>
#include <ctime>
#include <cmath>

using namespace std;

///       Funciones
/**
 * Luminosity: Algoritmo para convertir imagenes de color a grises; se basa en convertir el respectivo color (RGB) segun el espectro de vision humana.
   Este se expresa con que: Rojo(R)*0.21 | Verde(G)*0.71 | Azul(B)*0.07

 * Parametros:
       -Original_image: Imagen original, la cual a de ser convertida a escala de grises
       -gray_image: Imagen en blanco, en donde se almacenara la imagen convertida en escala de grises
       -max_x: Cantidad total de columnas (casillas en el eje X)
       -max_y Cantidad total de filas (casilas en el eje Y)
*/
void Luminosity(Mat Original_image, Mat gray_image, int max_x, int max_y)
{
    for(int x = 0; x < max_x; x++)
    {
        for(int y = 0; y < max_y; y++)
        {
            gray_image.at<Vec3b>(y,x)[0]=Original_image.at<Vec3b>(y,x)[0]*(0.21);
            gray_image.at<Vec3b>(y,x)[1]=Original_image.at<Vec3b>(y,x)[1]*(0.71);
            gray_image.at<Vec3b>(y,x)[2]=Original_image.at<Vec3b>(y,x)[2]*(0.07);
        }
    }
}

/**
 * Generar_mascara: Funcion encargada de preparar los valores internos de la mascara a utilizar en el difuminado a utilizar.

 * Parametros:
       -Base: Matriz bidimensional flotante la cual es utilizada como mascara para difuminado en otra funcion.
*/
void Generar_mascara(float base[5][5]){
    for(int i = 0; i<5; i++){
        for(int j = 0; j<5; j++){
            float expo = exp(-1*((pow(i-2,2)+pow(j-2,2))/(2*pow(1.5,2))));
            base[i][j]=expo/(2*3.1416*pow(1.5,2));
        }
    }
}

/**
 * Gaussian_blur: Algoritmo para difuminar una imagen; se basa en el metodo de difuminado gausiano, el cual utiliza una mascara sobre
   el area a difuminar para asi reducir el error relativo en los pixeles (se reduce la prob. de pixeles mal difuminados y problemas
   de iluminacion en la imagen)

 * Parametros:
       -Original_image: Imagen original, la cual a de ser convertida a escala de grises
       -gray_image: Imagen en blanco, en donde se almacenara la imagen convertida en escala de grises
       -max_x: Cantidad total de columnas (casillas en el eje X)
       -max_y Cantidad total de filas (casilas en el eje Y)
*/
void Gaussian_blur(Mat Original_image, Mat gray_image, int max_x, int max_y){
    float mascara[5][5]; ///Mascara flotante a utilizar
    Generar_mascara(mascara);
    for(int x=0; x<max_x; x++)
    {
        for(int y=0; y<max_y; y++)
        {
            for(int color=0; color<3; color++)
            {
                float sumador = 0;
                for(int xm=-2; xm<3; xm++)
                {
                    for(int ym=-2; ym<3; ym++)
                    {
                        if(xm+x>=0 && xm+x<max_x)
                        {
                            if(ym+y>=0 && ym+y<max_y)
                            {
                                sumador+=Original_image.at<Vec3b>(y+ym,x+xm)[color]*mascara[ym+2][xm+2];
                            }
                            else
                            {
                                sumador+=Original_image.at<Vec3b>(y,x+xm)[color]*mascara[ym+2][xm+2];
                            }
                        }
                        else
                        {
                            if(ky+y>=0 && ky+y<max_y)
                            {
                                sumador+=Original_image.at<Vec3b>(y+ym,x)[color]*mascara[ym+2][xm+2];
                            }
                            else
                            {
                                sumador+=Original_image.at<Vec3b>(y,x)[color]*mascara[ym+2][xm+2];
                            }
                        }
                    }
                }
                gray_image.at<Vec3b>(y,x)[color]=sumador;
            }
        }
    }
}

///       Main
int main()
{
    cout << "Hello world!" << endl;
    return 0;
}
