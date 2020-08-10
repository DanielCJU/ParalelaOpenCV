#include <iostream>
#include <cstdlib>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <omp.h>
#include <mpi.h>
#include <thread>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>

using namespace cv;
using namespace std;
int rangeMin, rangeMax;
/*
Titulo del Proyecto: Trabajo de imagenes

Descripcion del Proyecto:

Participantes: Ricardo Aliste G.
               Daniel Cajas U.
               Rodrigo Carmona R.
*/

/**-------------------------------------------------------------------- Funciones --------------------------------------------------------------------**/

/*
 * lineal_extrapolation: Funcion para realizar extrapolacion lineal
 * Parametros:
     -k1: Valor del punto actual
     -k0: valor del punto anterior al actual el K(n-1)
     -divi: Valor de la division por formula de k1 y k0 (buscar punto)
*/
float linear_extrapolation(float k1, float k0, float divi){
    return k1+(k0-k1)*divi;
}

/*
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

/*
*/
void obtener_fragmento(Mat imagen_original, Mat pedazo_recortado, int min_x, int min_y, int max_x, int max_y){
    for(int x=0; x<max_x-min_x; x++){
        for(int y=0; y<max_y-min_y; y++){
            pedazo_recortado.at<Vec3b>(y,x)[0]=imagen_original.at<Vec3b>(y,x+min_x)[0];
            pedazo_recortado.at<Vec3b>(y,x)[1]=imagen_original.at<Vec3b>(y,x+min_x)[1];
            pedazo_recortado.at<Vec3b>(y,x)[2]=imagen_original.at<Vec3b>(y,x+min_x)[2];
        }
    }
}

/*
*/
void join_gaussian_blur(Mat Original_image, Mat new_image, int proceso, int procesadores){
    int espaciado=(new_image.cols/procesadores)*proceso;
    int inicio=0, fin=0;
    if(proceso!=0){
        inicio=2;
    }
    if(proceso==procesadores-1){
        fin=-2;
    }
    for(int x=0; x<Original_image.cols+fin; x++){
        for(int y=0; y<Original_image.rows; y++){
            new_image.at<Vec3b>(y,espaciado+x)[0]=Original_image.at<Vec3b>(y,x+inicio)[0];
            new_image.at<Vec3b>(y,espaciado+x)[1]=Original_image.at<Vec3b>(y,x+inicio)[1];
            new_image.at<Vec3b>(y,espaciado+x)[2]=Original_image.at<Vec3b>(y,x+inicio)[2];
        }
    }
}

/*
*/
void join_luminosity_scale(Mat Original_image, Mat new_image,int proceso, int procesadores){
    int espaciado=(new_image.cols/procesadores)*proceso;
    for(int x=0; x<Original_image.cols; x++){
        for(int y=0; y<Original_image.rows; y++){
            new_image.at<Vec3b>(y,espaciado+x)[0]=Original_image.at<Vec3b>(y,x)[0];
            new_image.at<Vec3b>(y,espaciado+x)[1]=Original_image.at<Vec3b>(y,x)[1];
            new_image.at<Vec3b>(y,espaciado+x)[2]=Original_image.at<Vec3b>(y,x)[2];
        }
    }
}

/*
*/
void enviar(Mat imagen, int destinatario){
    size_t total, elemsize;
    int sizes[3];
    sizes[2] = imagen.elemSize();
    Size s = imagen.size();
    sizes[0] = s.height;
    sizes[1] = s.width;
    MPI_Send( sizes, 3, MPI_INT,destinatario,0,MPI_COMM_WORLD);
    MPI_Send( imagen.data, sizes[0]*sizes[1]*3, MPI_CHAR,destinatario,1, MPI_COMM_WORLD);
}

/*
*/
void recibir(Mat &fragmento,int remitente){
    MPI_Status estado;
    size_t total, elemsize;
    int sizes[3];
    MPI_Recv( sizes,3, MPI_INT,remitente,0, MPI_COMM_WORLD, &estado);
    fragmento.create(sizes[0], sizes[1], CV_8UC3);
    MPI_Recv( fragmento.data, sizes[0] * sizes[1] * 3, MPI_CHAR, remitente, 1, MPI_COMM_WORLD, &estado);
}

/*
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
                            if(ym+y>=0 && ym+y<max_y)
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

/*
 * Average: Algoritmo para convertir imagenes de color a grises; se basa en convertir el respectivo color (RGB) segun el espectro de vision humana.
   Este se expresa con que: Rojo(R)*0.21 | Verde(G)*0.71 | Azul(B)*0.07

 * Parametros:
       -Original_image: Imagen original, la cual a de ser convertida a escala de grises
       -gray_image: Imagen en blanco, en donde se almacenara la imagen convertida en escala de grises
       -max_x: Cantidad total de columnas (casillas en el eje X)
       -max_y Cantidad total de filas (casilas en el eje Y)
*/
void Average(Mat Original_image, Mat gray_image, int max_x, int max_y){
    float promedio;
    for(int x = 0; x < max_x; x++)
    {
        for(int y = 0; y < max_y; y++)
        {
            promedio=(Original_image.at<Vec3b>(y,x)[0]+Original_image.at<Vec3b>(y,x)[1]+Original_image.at<Vec3b>(y,x)[2])/3;
            gray_image.at<Vec3b>(y,x)[0]=promedio;
            gray_image.at<Vec3b>(y,x)[1]=promedio;
            gray_image.at<Vec3b>(y,x)[2]=promedio;
        }
    }
}


cv::Mat bi_lineal_scale(Mat imagen_original, float aumento){
    int columnas_nueva_imagen = imagen_original.cols*aumento;
    int filas_nueva_imagen = imagen_original.rows*aumento;
    Mat nueva_imagen(filas_nueva_imagen, columnas_nueva_imagen, CV_8UC3);
    float a1,a2;
    for(int x = 0; x < columnas_nueva_imagen; x++){
        for(int y = 0; y < filas_nueva_imagen; y++){
            float float_x = ((float)(x) / columnas_nueva_imagen) * (imagen_original.cols - 1);
            float float_y = ((float)(y) / filas_nueva_imagen) * (imagen_original.rows - 1);

            int int_x = (int) float_x;
            int int_y = (int) float_y;

            a1=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x)[0], imagen_original.at<Vec3b>(int_y+1, int_x)[0], float_x-int_x);
            a2=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x+1)[0], imagen_original.at<Vec3b>(int_y+1, int_x+1)[0], float_x-int_x);
            int R=linear_extrapolation(a1, a2, float_y-int_y);
            a1=0;
            a2=0;

            a1=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x)[1], imagen_original.at<Vec3b>(int_y+1, int_x)[1], float_x-int_x);
            a2=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x+1)[1], imagen_original.at<Vec3b>(int_y+1, int_x+1)[1], float_x-int_x);
            int G=linear_extrapolation(a1, a2, float_y-int_y);
            a1=0;
            a2=0;

            a1=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x)[2], imagen_original.at<Vec3b>(int_y+1, int_x)[2], float_x-int_x);
            a2=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x+1)[2], imagen_original.at<Vec3b>(int_y+1, int_x+1)[2], float_x-int_x);
            int B=linear_extrapolation(a1, a2, float_y-int_y);
            a1=0;
            a2=0;

            nueva_imagen.at<Vec3b>(y, x)[0] = R;
            nueva_imagen.at<Vec3b>(y, x)[1] = G;
            nueva_imagen.at<Vec3b>(y, x)[2] = B;
        }
    }
    return nueva_imagen;
}

int N_iteraciones(int filas, columnas)
{
     int pixeles=filas*columnas;
     if(pixeles<=163120)
     {
          return 1;
     }
     else
     {
          if(pixeles<=326240)
          {
               return 2;
          }
          else
          {
               if(pixeles<=1100710)
               {
                    return 4;
               }
               else
               {
                    return 6;
               }
          }
     }
}

int main(int argc, char** argv ){
    string option(argv[1]);
    Mat newimg;
    int iteraciones_blur=0;
    if(argc > 2){
        int mi_rango, procesadores;
        Mat img, fragmento, imagen_original;
        const auto hilos_posibles = std::thread::hardware_concurrency;
        std::cout<<"Hilos posibles: "<<hilos_posibles<<std::endl;
        if(hilos_posibles==0){
           std::cout<<"No se pudo identificar el hardware disponible"<<std::endl;
        } else {

           int provisto;
           MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provisto);
           if(provisto != MPI_THREAD_FUNNELED){
                std::cout<<"MPI ha entregado valor erroneo"<<std::endl;
           }
           MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);
           MPI_Comm_size(MPI_COMM_WORLD, &procesadores);

           //MPI_Init(&argc, &argv);
           //MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);
           //MPI_Comm_size(MPI_COMM_WORLD, &procesadores);
        }
        #pragma omp parallel default(none), \
                            shared(mi_rango), \
                            shared(procesadores), \
                            shared(option), \
                            shared(newimg), \
                            shared(img), \
                            shared(argv), \
                            shared(fragmento), \
                            shared(imagen_original), \
                            shared(ompi_mpi_comm_world), \
                            shared(ompi_mpi_int), \
                            shared(ompi_mpi_char)
        {
         #pragma omp master
         {
          if(mi_rango==0){
              string path(argv[2]);
              imagen_original=imread(path, 1);
              
              iteracion_blur=N_iteraciones(imagen_original.rows, imagen_original.cols);
              
              int diferencia=(imagen_original.cols/procesadores);
              int agregado=0;
              if(option=="1" || option=="2")
              {
                  agregado=2;
              }
              int mintemp=0, maxtemp=diferencia;

              Mat tmpfragmento(Size(diferencia+agregado, imagen_original.rows), imagen_original.type());
              fragmento = tmpfragmento.clone();
              obtener_fragmento(imagen_original, fragmento, 0, 0, diferencia+agregado, imagen_original.rows);

              for(int p=1; p<procesadores; p++){
                  mintemp=(diferencia*p)-agregado;
                  maxtemp=(diferencia*(p+1))+agregado;
                  if((p+1)==procesadores){
                      maxtemp=imagen_original.cols;
                  }
                  int diference=maxtemp-mintemp;
                  Mat imgToSend(Size(diference, imagen_original.rows), imagen_original.type());
                  obtener_fragmento(imagen_original, imgToSend, mintemp, 0, maxtemp, imagen_original.rows);
                  enviar(imgToSend, p);
              }
          } else {
              recibir(fragmento,0);
          }

          if(option=="3")
          {
               newimg.create(fragmento.rows*2, fragmento.cols*2, CV_8UC4);
          }
          else
          {
               newimg = fragmento.clone();
          }

          if(option=="1")
          {
              Gaussian_blur(fragmento, newimg, fragmento.cols, fragmento.rows);
              if(iteraciones_blur!=0)
              {
                   for(int i=0; i<iteraciones_blur; i++){Gaussian_blur(newimg, newimg, fragmento.cols, fragmento.rows);}
              }
              if(mi_rango == 0){
                  join_luminosity_scale(newimg, imagen_original, 0, procesadores);
                  for(int p = 1; p < procesadores; p++){
                      Mat imgtmpjoin;
                      recibir(imgtmpjoin, p);
                      join_gaussian_blur(imgtmpjoin, imagen_original, p, procesadores);
                  }
              } else {
                  enviar(newimg, 0);
              }
          }
          if(option == "2")
          {
              Average(fragmento, newimg, fragmento.cols, fragmento.rows);
              if(mi_rango == 0){
                  join_luminosity_scale(newimg, imagen_original, 0, procesadores);
                  for(int p = 1; p < procesadores; p++){
                      Mat imgtmpjoin;
                      recibir(imgtmpjoin, p);
                      join_luminosity_scale(imgtmpjoin, imagen_original, p, procesadores);
                  }
              } else {
                 enviar(newimg, 0);
              }
          }
          if(option == "3"){
              Mat tmpnewimg = bi_lineal_scale(fragmento, 2.0);
              if(mi_rango == 0){
                  //std::cout<<newimg.cols<<newimg.rows<<std::endl;
                  join_luminosity_scale(tmpnewimg, newimg, 0, procesadores);
                  for(int p = 1; p < procesadores; p++){
                      Mat imgtmpjoin;
                      recibir(imgtmpjoin, p);
                      join_luminosity_scale(imgtmpjoin, newimg, p, procesadores);
                  }
              } else {
                  enviar(tmpnewimg, 0);
              }
          }
     /*     if(option!="1" && option!="2" && option!="3"){
              std::cout<<"La opcion ingresada no es valida..."<<std::endl;
              return EXIT_FAILURE;
          }
          MPI_Finalize();
      } else {
          std::cout<<"No se ingresaron lo argumentos <opcion> <filepath>..."<<std::endl;
          return EXIT_FAILURE;
      }
     */
      }
     }
    }
    time_t now=time(0);
    struct tm tstruct;
    char buf[80];
    tstruct= *localtime(&now);
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tstruct);
    imwrite(option+"_"+string(buf)+".png", newimg);
    return EXIT_SUCCESS;
}
