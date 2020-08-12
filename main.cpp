#include <iostream>
#include <cstdlib>
#include <opencv4/opencv2/opencv_modules.hpp>
#include <opencv4/opencv2/photo.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core.hpp>
#include <mpi.h>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>

using namespace cv;
using namespace std;
int rangeMin, rangeMax;
/*------------------------------------------------------------ Funciones ------------------------------------------------------------*/

/*
 * N_iteraciones: Funcion que, en funcion de la cantidad total de pixeles, estima la cantidad de iteraciones de difuminado gausiano realizar.
 * Parametros:
     -filas: Cantidad de filas de la imagen
     -columnas: Cantidad de columnas de la imagen
*/
int N_iteraciones(int filas, int columnas)
{
     int pixeles=filas*columnas;
     if(pixeles<=40780)
     {
         return 2;
     }
     else
     {
         if(pixeles<=81560)
         {
              return 5;
         }
         else
         {
             if(pixeles<=122340)
             {
                 return 8;
             }
             else
             {
                 if(pixeles<=163120)
                 {
                      return 10;
                 }
                 else
                 {
                     if(pixeles<=356738)
                     {
                         return 14;
                     }
                     else
                     {
                         if(pixeles<=550355)
                         {
                              return 18;
                         }
                         else
                         {
                             return 22;
                         }
                    }
                }
            }
        }
    }
}

/*
 * lineal_extrapolation: Funcion para realizar extrapolacion lineal
 * Parametros:
     -k1: Valor del punto actual
     -k0: valor del punto anterior al actual el K(n-1)
     -divi: Valor de la division por formula de k1 y k0 (buscar punto)
*/
float linear_extrapolation(float k1, float k0, float divi)
{
    return k1+(k0-k1)*divi;
}

/*
 * Generar_mascara: Funcion encargada de preparar los valores internos de la mascara a utilizar en el difuminado a utilizar gausiano (Gaussian_blur).
 * Parametros:
       -Base: Matriz bidimensional flotante la cual es utilizada como mascara para difuminado en otra funcion.
*/
void Generar_mascara(float base[5][5]){
    for(int i = 0; i<5; i++)
    {
        for(int j = 0; j<5; j++)
        {
            float expo = exp(-1*((pow(i-2,2)+pow(j-2,2))/(2*pow(0.89,2))));
            base[i][j]=expo/(2*3.1416*pow(0.89,2));
        }
    }
}

/*
 * obtener_fragmento: Funcion encargada de generar fragmentos de una imagen; almacenando secciones de la imagen en funcion de los limites
 * Parametros:
       -imagen_original: Imagen original recibida, de esta se extraera el frgmento a almacenar.
       -pedazo_recortado: Imagen que contendra el fragmento de la imagen.
       -min_x y min_y: Limites inferiores del eje X y el eje _Y.
       -max_x y max_y: Limites superiores del eje X y el eje _Y.
*/
void obtener_fragmento(Mat imagen_original, Mat pedazo_recortado, int min_x, int min_y, int max_x, int max_y)
{
    for(int x=0; x<max_x-min_x; x++)
    {
        for(int y=0; y<max_y-min_y; y++)
        {
            pedazo_recortado.at<Vec3b>(y,x)[0]=imagen_original.at<Vec3b>(y,x+min_x)[0];
            pedazo_recortado.at<Vec3b>(y,x)[1]=imagen_original.at<Vec3b>(y,x+min_x)[1];
            pedazo_recortado.at<Vec3b>(y,x)[2]=imagen_original.at<Vec3b>(y,x+min_x)[2];
        }
    }
}

/*
 * join_gaussian_blur: Funcion encargada de unir todos los fragmentos tratados con difuminado gaussiano en los diversos procesos
 * Parametros:
       -Original_image: Imagen modificada a añadir.
       -new_image: Imagen donde almacenar todos los fragmentos ya procesados.
       -proceso: Numero del procesador recibido.
       -procesadores: Cantidad total de procesadores.
*/
void join_gaussian_blur(Mat Original_image, Mat new_image, int proceso, int procesadores)
{
    int espaciado=(new_image.cols/procesadores)*proceso;
    int inicio=0, fin=0;
    if(proceso!=0)
    {
        inicio=2;
    }
    if(proceso==procesadores-1)
    {
        fin=-2;
    }
    for(int x=0; x<Original_image.cols+fin; x++)
    {
        for(int y=0; y<Original_image.rows; y++)
        {
            new_image.at<Vec3b>(y,espaciado+x)[0]=Original_image.at<Vec3b>(y,x+inicio)[0];
            new_image.at<Vec3b>(y,espaciado+x)[1]=Original_image.at<Vec3b>(y,x+inicio)[1];
            new_image.at<Vec3b>(y,espaciado+x)[2]=Original_image.at<Vec3b>(y,x+inicio)[2];
        }
    }
}



/*
 * join_luminosity_scale: Funcion encargada de unir todos los fragmentos tratados con el re-escalado, o escalado de grises en los diversos procesos
 * Parametros:
       -Original_image: Imagen modificada a añadir.
       -new_image: Imagen donde almacenar todos los fragmentos ya procesados.
       -proceso: Numero del procesador recibido.
       -procesadores: Cantidad total de procesadores.
*/
void join_luminosity_scale(Mat Original_image, Mat new_image,int proceso, int procesadores)
{
    int espaciado=(new_image.cols/procesadores)*proceso;
    for(int x=0; x<Original_image.cols; x++)
    {
        for(int y=0; y<Original_image.rows; y++)
        {
            new_image.at<Vec3b>(y,espaciado+x)[0]=Original_image.at<Vec3b>(y,x)[0];
            new_image.at<Vec3b>(y,espaciado+x)[1]=Original_image.at<Vec3b>(y,x)[1];
            new_image.at<Vec3b>(y,espaciado+x)[2]=Original_image.at<Vec3b>(y,x)[2];
        }
    }
}

/*
 * enviar: Funcion encargada de realizar el envio de los fragmentos y la comunicacion de envio
 * Parametros:
       -imagen: Imagen a enviar.
       -destinatario: Numero del proceso enviante.
*/
void enviar(Mat imagen, int destinatario)
{
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
 * enviar: Funcion encargada de realizar el envio de los fragmentos y la comunicacion de envio
 * Parametros:
       -fragmento: Imagen a enviar.
       -remitente: Numero del procesador recibido.
*/
void recibir(Mat &fragmento,int remitente)
{
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
void Gaussian_blur(Mat Original_image, Mat gray_image, int max_x, int max_y)
{
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
 * Average: Algoritmo para convertir imagenes de color a grises; se basa en promediar los valores de los 3 canales (RGB) del respectivo pixeles
 * Parametros:
       -Original_image: Imagen original, la cual a de ser convertida a escala de grises
       -gray_image: Imagen en blanco, en donde se almacenara la imagen convertida en escala de grises
       -max_x y max_y: Cantidad total de columnas (X) y de filas (Y)
*/
void Average(Mat Original_image, Mat gray_image, int max_x, int max_y)
{
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

/*
 * bi_lineal_scale: Algoritmo para re escalado de imagenes; se basa en en aplicar una extrapolacion bilineal (osea, una extrapolacion lineal a otras 
 2 extrapolaciones lineales), para generar el pixel correspondiente a la ubicacion.
 * Parametros:
       -Original_image: Imagen original, la cual a de ser re escalada
       -nueva_imagen: Imagen en blanco, en donde se almacenara la imagen escalada
       -aumento: Valor por el cual la imagen sera re-escalada (x2)
*/
void bi_lineal_scale(Mat imagen_original, Mat nueva_imagen, int aumento)
{
    float L1, L2;
    for(int x = 0; x < imagen_original.cols*aumento; x++){
        for(int y = 0; y < imagen_original.rows*aumento; y++){
            float float_x = ((float)(x) / imagen_original.cols*aumento) * (imagen_original.cols - 1);
            float float_y = ((float)(y) / imagen_original.rows*aumento) * (imagen_original.rows - 1);

            int int_x = (int) float_x;
            int int_y = (int) float_y;
            L1=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x)[0], imagen_original.at<Vec3b>(int_y + 1, int_x)[0], float_x - int_x);
            L2=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x + 1)[0], imagen_original.at<Vec3b>(int_y + 1, int_x + 1)[0], float_x - int_x);
            int R = linear_extrapolation(L1, L2, float_y - int_y);
            
            L1=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x)[1], imagen_original.at<Vec3b>(int_y + 1, int_x)[1], float_x - int_x);
            L2=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x + 1)[1], imagen_original.at<Vec3b>(int_y + 1, int_x + 1)[1], float_x - int_x);
            int G = linear_extrapolation(L1, L2, float_y - int_y);
            
            L1=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x)[2], imagen_original.at<Vec3b>(int_y + 1, int_x)[2], float_x - int_x);
            L2=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x + 1)[2], imagen_original.at<Vec3b>(int_y + 1, int_x + 1)[2], float_x - int_x);
            int B = linear_extrapolation(L1, L2, float_y - int_y);

            nueva_imagen.at<Vec3b>(y, x)[0] = R;
            nueva_imagen.at<Vec3b>(y, x)[1] = G;
            nueva_imagen.at<Vec3b>(y, x)[2] = B;
        }
    }
}


/*------------------------------------------------------------   Main   ------------------------------------------------------------*/
int main(int argc, char** argv ){
    string option(argv[1]);
    Mat newimg;
    int iteraciones_blur=0;
    if(argc > 2){
        int mi_rango, procesadores;
        Mat img, fragmento, imagen_original;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);
        MPI_Comm_size(MPI_COMM_WORLD, &procesadores);

        if(mi_rango==0){
            string path(argv[2]);
            imagen_original=imread(path, 1);
             
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
        }
        else{
            recibir(fragmento,0);
        }
        if(option=="1" || option=="2")
        {
             newimg = fragmento.clone();
        }
        else{
             newimg.create(fragmento.rows*2, fragmento.cols*2, CV_8UC3);
        }
         if(option=="1")
        {
            if(iteraciones_blur==0){
                iteraciones_blur=N_iteraciones(fragmento.rows, fragmento.cols);
            }
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
                      newimg=imagen_original.clone();
                  }
              }
              else{
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
                     newimg=imagen_original.clone();
                }
            }
            else{
                enviar(newimg, 0);
            }
        }
        if(option == "3"){
            Mat tmpnewimg;
            bi_lineal_scale(fragmento, tmpnewimg, 2.0);
            cout<<tmpnewimg.cols<<"-"<<tmpnewimg.cols<<endl;
            if(mi_rango == 0){
                Mat img_escalada(imagen_original.rows*2, imagen_original.cols*2, CV_8UC3);
                join_luminosity_scale(tmpnewimg, img_escalada, 0, procesadores);
                for(int p = 1; p < procesadores; p++){
                    Mat imgtmpjoin;
                    recibir(imgtmpjoin, p);
                    cout<<"re:"<<imgtmpjoin.cols<<"-"<<imgtmpjoin.cols<<endl;
                    join_luminosity_scale(imgtmpjoin, imagen_original, p, procesadores);
                    newimg=imagen_original.clone();
                }
            }
            else{
                enviar(tmpnewimg, 0);
            }
        }
        if(option!="1" && option!="2" && option!="3"){
            cout<<"La opcion ingresada no es valida..."<<endl;
            return EXIT_FAILURE;
        }
        MPI_Finalize();
    }
    else{
        cout<<"No se ingresaron lo argumentos <opcion> <filepath>..."<<endl;
        return EXIT_FAILURE;
    }
    time_t now=time(0);
    struct tm tstruct;
    char buf[80];
    tstruct= *localtime(&now);
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tstruct);
    imwrite(option+"_"+string(buf)+".png", newimg);
    return EXIT_SUCCESS;
}
