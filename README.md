# ParalelaOpenCV

_Proyecto de imágenes computación paralela y distribuida con OpenCV y OpenMPI(Trabajo 3)_

### Pre-requisitos

_Sistema operativo soportado:_

* OS: linux (testeado en Ubuntu 20.4)

_Librerias necesarias:_
* Compilador GNU C++ y librerías básicas de C/C++: [sudo apt-get install build-essentials]
* git, para clonar el repositorio: [sudo apt-get install git-core]
* Openmpi: [sudo apt-get install libopenmpi-dev]
* ssh, para utilizar varias máquinas en cluster: [sudo apt-get install openssh-server]
* cmake, para configurar la compilación del programa: [sudo apt-get install cmake]
* OpenCV 4 o superior

### Utilización
Pasos para utilizar el programa:

1) Construir el archivo Makefile y demás configuraciones del programa con cmake. Para esto se debe primero alterar la ruta de instalación de OpenCV en CMakeLists.txt (línea 11) si fuera necesario. Luego:
- Crear una carpeta de construcción y situarse en ésta por terminal.
- Utilizar cmake desde la carpeta de construcción.
_Ejemplo:_

```
/home/master$ mkdir build && cd build
/home/master/build$ cmake /home/master
```

2) Compilar el ejecutable con el archivo Makefile. Esto se hace con el comando make.

```
/home/master/build$ make
```

3) Ejecutar el programa con la siguiente sintaxis: mpirun ./OCVTest [número de operación a realizar] [ruta de la imagen a procesar].

```
/home/master/build$ mpirun ./OCVTest 1 /home/img/motor.jpeg
/home/master/build$ mpirun ./OCVTest 2 /home/img/control.png
```
### Detalles

El programa cuenta con tres operaciones: difuminar (opción número 1), pasar a escala de grises (opción número 2) y escalado de 2.0 (opción número 3). El resultado de cada una de las operaciones es una imágen que se crea en la carpeta de ejecución del programa. La nomenclatura del archivo es N°operación_fecha.png, siendo el formato de la fecha yyyyMMddhhmmss. Ejemplo:

```
1_20200808204120.png
```
### Acotaciones Particulares

* El programa se testeo con 1 y 2 procesadores, debido a limitaciones de hardware.
* Debido al funcionamiento paralelo de MPI, en ocasiones el programa entraga imagenes en blanco, o mas de una imagen para algun tratamiento; en caso de suceder, se recomienda volver a realizar la ejecucion.

### Equipo Desarrollador:
* Ricardo Aliste G. Desarrollador/Documentador
* Daniel Cajas U. Desarrollador/Documentador
* Rodrigo Carmona R. Documentador
