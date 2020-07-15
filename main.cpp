#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <stdio.h>
#include "mpi.h"
#include <thread>
#include <cstdlib>
#include <chrono>

using namespace cv;
using namespace std;

char* tiempoActual(char* buffer){
    std::time_t datos;
    std::tm* info;
    std::time(&datos);
    info = std::localtime(&datos);

    std::strftime(buffer,80,"%Y%m%d%H%M%S",info);

    return buffer;
}

std::string convertirEnString(char* arreglo, int largo){ 
    string texto = ""; 
    for (int i = 0; i < largo; i++) { 
        texto = texto + arreglo[i]; 
    } 
    return texto;
}

int convertirStringEnEntero(std::string linea){
    try{
        int entero = stoi(linea);
        return entero;
    }
    catch(std::invalid_argument const &e){
        std::cout << "Argumento inválido" << std::endl;
        return 666;
    }
}

void matsnd(const Mat& m,int dest){
      int rows  = m.rows;
      int cols  = m.cols;
      int type  = m.type();
      int channels = m.channels();
      memcpy(&buffer[0 * sizeof(int)],(uchar*)&rows,sizeof(int));
      memcpy(&buffer[1 * sizeof(int)],(uchar*)&cols,sizeof(int));
      memcpy(&buffer[2 * sizeof(int)],(uchar*)&type,sizeof(int));
	  cout << "matsnd: filas=" << rows << endl;
      cout << "matsnd: columnas=" << cols << endl;
      cout << "matsnd: type=" << type << endl;
      cout << "matsnd: channels=" << channels << endl;
      cout << "matsnd: bytes=" << bytes << endl;

      int bytespersample=1;
      int bytes=m.rows*m.cols*channels*bytespersample;
	  
      if(!m.isContinuous())
      { 
         m = m.clone();
      }
      memcpy(&buffer[3*sizeof(int)],m.data,bytes);
      MPI_Send(&buffer,bytes+3*sizeof(int),MPI_UNSIGNED_CHAR,dest,0,MPI_COMM_WORLD);
}

Mat matrcv(int src){
      MPI_Status status;
      int count,rows,cols,type,channels;

      MPI_Recv(&buffer,sizeof(buffer),MPI_UNSIGNED_CHAR,src,0,MPI_COMM_WORLD,&status);
      MPI_Get_count(&status,MPI_UNSIGNED_CHAR,&count);
      memcpy((uchar*)&rows,&buffer[0 * sizeof(int)], sizeof(int));
      memcpy((uchar*)&cols,&buffer[1 * sizeof(int)], sizeof(int));
      memcpy((uchar*)&type,&buffer[2 * sizeof(int)], sizeof(int));

      cout << "matrcv: Count=" << count << endl;
      cout << "matrcv: filas=" << rows << endl;
      cout << "matrcv: columnas=" << cols << endl;
      cout << "matrcv: type=" << type << endl;

      Mat received= Mat(rows,cols,type,(uchar*)&buffer[3*sizeof(int)]);
      return received;
}

int main( int argc, char** argv )
{
    int mi_rango; /* rango del proceso    */
    int procesadores; /* numero de procesos   */
    int maestro = 0; /* Identificador maestro */
    MPI_Status estado; /* devuelve estado al recibir*/
	
	/* Este string se usara para detener los hilos paralelos */
    std::string parar("STOP");
	
    if(argc!=3){
        std::cout << "No se han entregado los argumentos necesarios. Se cerrará el programa" << std::endl;
        return EXIT_FAILURE;
    }
        
	/* Comienza las llamadas a MPI */
    MPI_Init(&argc, &argv);

    /* Averiguamos el rango de nuestro proceso */
    MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);

    /* Averiguamos el número de procesos que estan 
    * ejecutando nuestro porgrama 
    */
    MPI_Comm_size(MPI_COMM_WORLD, &procesadores);
	
    if (procesadores < 2) {
            fprintf(stderr, "\nLa implementación requiere al menos 2 procesadores\n");
            return EXIT_FAILURE;
    }
		
    std::string opcion(argv[1]);
    std::cout << opcion << std::endl;
    int numero = convertirStringEnEntero(opcion);

    if(numero==666){
        return EXIT_FAILURE;
    }
    
    if(numero==1){
	   if(mi_rango==0){
		   Mat received=matrcv(1);
		   char temporalUno[13];
                   std::string NombreUno = "operacion_1_"+convertirEnString(tiempoActual(temporalUno), 14)+".png";
		   imwrite("NombreUno",received);
		   received.release();
		   return 0;
	   } else {
		   std::string ruta(argv[2]);
		   cv::Mat imagenOriginal = imread(ruta);
		   if (!imagenOriginal.data) {
                        return EXIT_FAILURE;
           }
           cv::Mat imagenDifuminada;
           cv::GaussianBlur(imagenOriginal, imagenDifuminada, cv::Size(3, 3), 0);
           imagenOriginal.release();
           matsnd(imagenDifuminada,0);
	   }
    }

    if(numero==2){
        cv::Mat imagenGris;
        cv::cvtColor(imagenOriginal, imagenGris, COLOR_BGR2GRAY);
        imagenOriginal.release();
        char temporalDos[13];
        std::string NombreDos = "operacion_2_"+convertirEnString(tiempoActual(temporalDos), 14)+".png";
        imwrite(NombreDos, imagenGris);
        imagenGris.release();
        return 0;
    }
    if(numero==3){
        cv::Mat imagenEscalada;
        cv::resize(imagenOriginal, imagenEscalada, cv::Size(), 1.5, 1.5);
        imagenOriginal.release();
        char temporalTres[13];
        std::string NombreTres = "operacion_3_"+convertirEnString(tiempoActual(temporalTres), 14)+".png";
        imwrite(NombreTres, imagenEscalada);
        imagenEscalada.release();
        return 0;
    }
    return 1;
}
