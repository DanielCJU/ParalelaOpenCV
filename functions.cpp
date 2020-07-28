#include "functions.h"
using namespace cv;
using namespace std;
int rangeMin, rangeMax;

/** Funciones **/

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
 * bi_linear_extrapolation: Funcion encargada de realizar extrapolacion bilinear, mediante 2 extrapolaciones lineales
 *Parametros:
     -a: Punto x1
     -b: Punto y1
     -c: Punto x2
     -d: Punto y2
     -x: Valor division para Punto 1
     -y: Valor division para Punto 2
*/
float bi_linear_extrapolation(float a, float b, float c, float d, float x, float y){
    return linear_extrapolation(linear_extrapolation(a, b, x), linear_extrapolation(c, d, x), y);
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
    int newcols = imagen_original.cols*aumento;
    int newrows = imagen_original.rows*aumento;
    Mat nueva_imagen(newrows, newcols, CV_8UC3);
    for(int x = 0; x < newcols; x++){
        for(int y = 0; y < newrows; y++){
            float gx = ((float)(x) / newcols) * (imagen_original.cols - 1);
            float gy = ((float)(y) / newrows) * (imagen_original.rows - 1);

            int gxi = (int) gx;
            int gyi = (int) gy;
            int red = bi_linear_extrapolation(imagen_original.at<Vec3b>(gyi, gxi)[0], imagen_original.at<Vec3b>(gyi + 1, gxi)[0], imagen_original.at<Vec3b>(gyi, gxi + 1)[0], imagen_original.at<Vec3b>(gyi + 1, gxi + 1)[0], gx - gxi, gy - gyi);
            int green = bi_linear_extrapolation(imagen_original.at<Vec3b>(gyi, gxi)[1], imagen_original.at<Vec3b>(gyi + 1, gxi)[1], imagen_original.at<Vec3b>(gyi, gxi + 1)[1], imagen_original.at<Vec3b>(gyi + 1, gxi + 1)[1], gx - gxi, gy - gyi);
            int blue = bi_linear_extrapolation(imagen_original.at<Vec3b>(gyi, gxi)[2], imagen_original.at<Vec3b>(gyi + 1, gxi)[2], imagen_original.at<Vec3b>(gyi, gxi + 1)[2], imagen_original.at<Vec3b>(gyi + 1, gxi + 1)[2], gx - gxi, gy - gyi);

            nueva_imagen.at<Vec3b>(y, x)[0] = red;
            nueva_imagen.at<Vec3b>(y, x)[1] = green;
            nueva_imagen.at<Vec3b>(y, x)[2] = blue;
        }
    }
    return nueva_imagen;
}