#include "functions.h"
using namespace cv;
using namespace std;
int rangeMin, rangeMax;

int main(int argc, char** argv ){
    string option(argv[1]);
    Mat newimg;
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
        newimg = fragmento.clone();
        if(option=="1")
        {
            Gaussian_blur(fragmento, newimg, fragmento.cols, fragmento.rows);
            if(mi_rango == 0){
                join_luminosity_scale(newimg, imagen_original, 0, procesadores);
                for(int p = 1; p < procesadores; p++){
                    Mat imgtmpjoin;
                    recibir(imgtmpjoin, p);
                    join_gaussian_blur(imgtmpjoin, imagen_original, p, procesadores);
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
                }
            }
            else{
                enviar(newimg, 0);
            }
        }
        if(option == "3"){
            Mat tmpnewimg = bi_lineal_scale(fragmento, 2.0);
            if(mi_rango == 0){
                Mat newimg2(imagen_original.rows*2, imagen_original.cols*2, CV_8UC3);
                cout<<newimg.cols<<newimg.rows<<endl;
                join_luminosity_scale(tmpnewimg, newimg, 0, procesadores);
                for(int p = 1; p < procesadores; p++){
                    Mat imgtmpjoin;
                    recibir(imgtmpjoin, p);
                    join_luminosity_scale(imgtmpjoin, newimg2, p, procesadores);
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
    if(option=="3")
    {
        imwrite(option+"_"+string(buf)+".png", newimg2);
    }
    else
    {
        imwrite(option+"_"+string(buf)+".png", newimg);
    }
    return EXIT_SUCCESS;
}
