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
/*------------------------------------------------------------ Datos Clave ------------------------------------------------------------*/
/*
Titulo Proyecto:Proyecto de Imagenes Paralelas

Equipo Desarrollador:
     -Ricardo Aliste G.
     -Daniel Cajas U.
     -Rodrigo Carmona R.
     
Descripcion del Proyecto:
     El proyecto consta de realizar un programa, el cual mediante el recibimiento de cierta informacion (direccion de una imagen, 
     opcion a realizar dentro del sistema, etc) efectue cierto tipo de tratamiento a una imagen; son 3 posibles tratamientos, los 
     cuales son:
     
     -Difuminado Gaussiano: Realiza un difuminado de la imagen mediante el algoritmo gaussiano, el cual trabaja usando una mascara 
     de 5X5; este difuminado se aplica una cierta cantidad de veces, dependiendo de la cantidad de pixeles que componen la imagen.
     
     -Transformacion a Escala de Grises: Se realiza la transformacion de la imagen a su equivalente, pero en escala de grises; para 
     esto, se ace uso del algoritmo de promediacion (o Average, por su nombre en ingles), el cual consta de promediar el valor del 
     canal rojo (R), verde (G) y azul (B), y asignar ese valor a los 3 canales.
     
     -Re-escalado: Se realiza un aumento de tamaño de la imagen (del doble por defecto); Esto se realiza mediante el algoritmo de 
     Extrapolacion Bilineal, la cual se resume en efectuar una extrapolacion lineal a 2 extrapolaciones lineales, las cuales se 
     generan en funcion de las coordenadas colindantes, y las diferencia de las coordenadas actuales y las resultantes.
*/

/*------------------------------------------------------------ Funciones ------------------------------------------------------------*/

/*
 * N_iteraciones: Funcion que, en funcion de la cantidad total de pixeles, estima la cantidad de iteraciones de difuminado gausiano realizar.
 * Parametros:
     -filas: Cantidad de filas de la imagen
     -columnas: Cantidad de columnas de la imagen
     -procesadores: Cantidad total de procesadores
*/
int N_iteraciones(int filas, int columnas, int procesadores)
{
     int pixeles=filas*columnas;
     if(pixeles<=(40780*2)/procesadores) ///Casos de Imagenes relativamente mas pequeñas
     {
         return 2;
     }
     else
     {
         if(pixeles<=((81560*2)/procesadores)) ///Casos de Imagenes relativamente mas pequeñas
         {
              return 5;
         }
         else
         {
             if(pixeles<=((122340*2)/procesadores)) ///Casos de Imagenes relativamente "Medianas"
             {
                 return 8;
             }
             else
             {
                 if(pixeles<=((163120*2)/procesadores)) ///Casos de Imagenes relativamente "Medianas"
                 {
                      return 10;
                 }
                 else
                 {
                     if(pixeles<=((356737.5*2)/procesadores)) ///Casos de Imagenes relativamente Grandes
                     {
                         return 14;
                     }
                     else
                     {
                         if(pixeles<=((550355*2)/procesadores)) ///Casos de Imagenes relativamente Grandes
                         {
                              return 18;
                         }
                         else ///Casos de Imagenes relativamente muy grandes
                         {
                             return ((22*2)/procesadores); 
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
       
 Enlace a formula para mayor referencias: https://wikimedia.org/api/rest_v1/media/math/render/svg/6717136818f2166eba2db0cfc915d732add9c64f
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
    Generar_mascara(mascara); ///Se rellena la mascara
    for(int x=0; x<max_x; x++) 
    {
        for(int y=0; y<max_y; y++) ///Se realiza doble ciclo FOR para recorer toda la matriz de la imagen
        {
            for(int color=0; color<3; color++) ///Se efectua analisis Para cada canal (RGB)
            {
                float sumador = 0;
                for(int xm=-2; xm<3; xm++) ///Se realiza analisis por cada cordenada dentro de los limites de la mascara
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
                gray_image.at<Vec3b>(y,x)[color]=sumador; ///Se registra el valor obtenido
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
        for(int y = 0; y < max_y; y++) ///Se realiza doble ciclo FOR para recorrer matriz
        {
            promedio=(Original_image.at<Vec3b>(y,x)[0]+Original_image.at<Vec3b>(y,x)[1]+Original_image.at<Vec3b>(y,x)[2])/3; ///Se promedian los valores de los 3 canales
            gray_image.at<Vec3b>(y,x)[0]=promedio; ///Se procede a asignar el valor a cada canal
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
cv::Mat bi_lineal_scale(Mat imagen_original, float aumento){
    int columnas_nueva_imagen = imagen_original.cols*aumento;
    int filas_nueva_imagen = imagen_original.rows*aumento;
    Mat nueva_imagen(filas_nueva_imagen, columnas_nueva_imagen, CV_8UC3); ///Se genera imagen donde se almacenara la imagen escalda
    float a1,a2; ///Variables de apoyo, son para almacenar las extrapolaciones lineales
    for(int x = 0; x < columnas_nueva_imagen; x++)
    {
        for(int y = 0; y < filas_nueva_imagen; y++) ///Doble ciclo FOR para recorrer matriz
        {
            float float_x = ((float)(x) / columnas_nueva_imagen) * (imagen_original.cols - 1);
            float float_y = ((float)(y) / filas_nueva_imagen) * (imagen_original.rows - 1);
             
            int int_x = (int) float_x;
            int int_y = (int) float_y;
             
            float k1=float_x-int_x; ///Divisores a usar en las extrapolacones lineales 
            float k2=float_y-int_y;
             
            a1=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x)[0], imagen_original.at<Vec3b>(int_y+1, int_x)[0], k1); ///Se efectuan 2 extrapolaciones lineales
            a2=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x+1)[0], imagen_original.at<Vec3b>(int_y+1, int_x+1)[0], k1);
            int R=linear_extrapolation(a1, a2, k2); ///Se efectua la extrapolacion lineal de las 2 anteriores (extrapolacion bilineal); se efectua esto para cada canal
             
            a1=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x)[1], imagen_original.at<Vec3b>(int_y+1, int_x)[1], k1);
            a2=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x+1)[1], imagen_original.at<Vec3b>(int_y+1, int_x+1)[1], k1);
            int G=linear_extrapolation(a1, a2, k2);
             
            a1=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x)[2], imagen_original.at<Vec3b>(int_y+1, int_x)[2], k1);
            a2=linear_extrapolation(imagen_original.at<Vec3b>(int_y, int_x+1)[2], imagen_original.at<Vec3b>(int_y+1, int_x+1)[2], k1);
            int B=linear_extrapolation(a1, a2, k2);
             
            nueva_imagen.at<Vec3b>(y, x)[0] = R; ///Se realiza almacenado en cada canal
            nueva_imagen.at<Vec3b>(y, x)[1] = G;
            nueva_imagen.at<Vec3b>(y, x)[2] = B;
        }
    }
    return nueva_imagen; ///Se retorna imagen resultante
}

/*------------------------------------------------------------   Main   ------------------------------------------------------------*/
int main(int argc, char** argv ){
    string option(argv[1]);
    
    Mat newimg, img; ///Variables de imagenes para respuesta final y apoyo
    int iteraciones_blur=0; ///Variable para definir cantidad de iteraciones de difuminado gaussiano
    
    if(argc > 2) ///En caso de recibir mas de 2 parametros (escenario esperado)
    { 
        int mi_rango, procesadores;
        Mat fragmento, imagen_original;
        
        ///Activacion MPI
        MPI_Init(&argc, &argv); 
        MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);
        MPI_Comm_size(MPI_COMM_WORLD, &procesadores);
        
        if(mi_rango==0) ///Caso de configuracion inicial
        { 
            string path(argv[2]); ///Se obtiene el PATH de la imagen
            imagen_original=imread(path, 1); ///Se almacena la imagen a tratar
             
            int diferencia=(imagen_original.cols/procesadores); ///Se detecta tamaño de las imagenes para proceder a cortarlas
            int agregado=0;
            if(option=="1" || option=="2") ///Se le añade un marco adicional debido a limites de la imagen, y por tratamiento del difuminado
            {
                agregado=2;
            }
            int mintemp=0, maxtemp=diferencia;

            Mat tmpfragmento(Size(diferencia+agregado, imagen_original.rows), imagen_original.type()); ///Se genera imagen que almacena fragmento a tratar de la imagen
            fragmento = tmpfragmento.clone();
            obtener_fragmento(imagen_original, fragmento, 0, 0, diferencia+agregado, imagen_original.rows);

            for(int p=1; p<procesadores; p++)
            {
                mintemp=(diferencia*p)-agregado;
                maxtemp=(diferencia*(p+1))+agregado;
                if((p+1)==procesadores)
                {
                    maxtemp=imagen_original.cols;
                }
                int diference=maxtemp-mintemp;
                Mat imgToSend(Size(diference, imagen_original.rows), imagen_original.type());
                obtener_fragmento(imagen_original, imgToSend, mintemp, 0, maxtemp, imagen_original.rows);
                enviar(imgToSend, p);
            }
        }
        else
        {
            recibir(fragmento,0);
        }
        
        ///Configuraciones para almacenamiento posterior de las imagenes
        if(option=="1" || option=="2") ///Caso de difuminado y escala de grises
        {
             newimg = fragmento.clone();
        }
        else ///Caso de Re-escalado
        {
             newimg.create(fragmento.rows*2, fragmento.cols*2, CV_8UC3);
        }
         if(option=="1") ///Difuminado Gaussiano
        {
            if(iteraciones_blur==0)
            {
                iteraciones_blur=N_iteraciones(fragmento.rows, fragmento.cols, procesadores); ///Se estima la cantidad optima de iteraciones de difuminado
            }
            Gaussian_blur(fragmento, newimg, fragmento.cols, fragmento.rows); ///Se efectua el primer difuminado
              
            if(iteraciones_blur!=0) ///Encaso de que se considere necesario efectuar mas difuminados
            {
                for(int i=0; i<iteraciones_blur; i++){Gaussian_blur(newimg, newimg, fragmento.cols, fragmento.rows);} ///Se realiza ciclo FOR de difuminados
            }
            if(mi_rango == 0) ///Caso de ser el proceso principal (se repite en las demas opciones)
            { 
                  join_luminosity_scale(newimg, imagen_original, 0, procesadores); ///Se realiza union con imagen final
                  for(int p = 1; p < procesadores; p++) ///Ciclo para unir todos los procesos
                  {
                      Mat imgtmpjoin;
                      recibir(imgtmpjoin, p); ///Se recibe imagen tratada
                      join_gaussian_blur(imgtmpjoin, imagen_original, p, procesadores); ///Se realiza union de estas
                      newimg=imagen_original.clone(); ///Se almacena la imagen resultante
                  }
              }
              else ///Caso de no serlo (se repite en las demas opciones)
              {
                  enviar(newimg, 0); ///Se envia imagen tratada
              }
        }
        if(option == "2") ///Caso de escala de grises
        {
            Average(fragmento, newimg, fragmento.cols, fragmento.rows); ///Se realiza transformacion de colores a escala de grises
            if(mi_rango == 0) ///Caso de ser el proceso principal
            {
                join_luminosity_scale(newimg, imagen_original, 0, procesadores);
                for(int p = 1; p < procesadores; p++)
                {
                    Mat imgtmpjoin;
                    recibir(imgtmpjoin, p);
                    join_luminosity_scale(imgtmpjoin, imagen_original, p, procesadores);
                     newimg=imagen_original.clone();
                }
            }
            else ///Caso de no serlo
            {
                enviar(newimg, 0);
            }
        }
        if(option == "3") ///Caso de re-escalado bilineal
        {
             Mat tmpnewimg = bi_lineal_scale(fragmento, 2.0); ///Se realiza y almacena imagen re escalada
             if(mi_rango == 0)///Caso de ser el proceso principal
             {
                 Mat newimg2(imagen_original.rows*2, imagen_original.cols*2, CV_8UC3);
                 join_luminosity_scale(tmpnewimg, newimg2, 0, procesadores);
                 for(int p = 1; p < procesadores; p++)
                 {
                     Mat imgtmpjoin;
                     recibir(imgtmpjoin, p);
                     join_luminosity_scale(imgtmpjoin, newimg2, p, procesadores);
                     newimg=newimg2.clone();
                 }
             }
             else ///Caso de no serlo
             {
                 enviar(tmpnewimg, 0);
             }
        }
        if(option!="1" && option!="2" && option!="3")
        {
            cout<<"La opcion ingresada no es valida; porfavor ingresar valores 1, 2 o 3"<<endl;
            return EXIT_FAILURE; ///Salida en caso de EXCEPCION
        }
         
        MPI_Finalize(); ///Finalizar MPI
    }
     
    else ///Caso en el que la cantidad de parametros sea menor o igual a 2
    {
        cout<<"No se ingresaron lo argumentos necesarios"<<endl;
        return EXIT_FAILURE; ///Salida en caso de EXCEPCION
    }
    
    
    time_t now=time(0); ///Se obtiene La hora actual
    struct tm tstruct; ///Variable para lamacenar la fecha actual
    char buf[80]; ///Variable para lamacenar la fecha actual
    tstruct= *localtime(&now); ///Se obtiene tiempo actual
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tstruct); ///Se Registra fecha actual
    imwrite(option+"_"+string(buf)+".png", newimg); ///Se almacena la imagen resultante
    return EXIT_SUCCESS; ///Fin del programa
}
