#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <stdio.h>

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
int main( int argc, char** argv )
{
    if(argc!=3){
        std::cout << "No se han entregado los argumentos necesarios. Se cerrará el programa" << std::endl;
        return EXIT_FAILURE;
    }

    std::string ruta(argv[2]);
    std::string opcion(argv[1]);
    std::cout << opcion << std::endl;
    int numero = convertirStringEnEntero(opcion);

    if(numero==666){
        return EXIT_FAILURE;
    }

    std::cout << ruta << std::endl;
    cv::Mat imagenOriginal = imread(ruta);
    
    if (!imagenOriginal.data) {
        return EXIT_FAILURE;
    }
    
    if(numero==1){
       cv::Mat imagenDifuminada;
       cv::GaussianBlur(imagenOriginal, imagenDifuminada, cv::Size(3, 3), 0);
       imagenOriginal.release();
       char temporalUno[13];
       std::string NombreUno = "operacion_1_"+convertirEnString(tiempoActual(temporalUno), 14)+".png";
       imwrite(NombreUno, imagenDifuminada);
       imagenDifuminada.release();
       return 0; 
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
    return 0;
}
