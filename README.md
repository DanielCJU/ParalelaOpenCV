# ParalelaOpenCV
Pasos para utilizar el programa:
1) Construir el archivo Makefile y demás configuraciones del programa con cmake. Para esto se debe:
- Alterar la ruta de instalación de OpenCV en CMakeLists.txt si fuera necesario.
- Crear una carpeta de construcción y situarse en ésta por terminal. Por ejemplo: mkdir build && cd build.
- Utilizar cmake desde la carpeta de construcción. Por ejemplo: cmake [ruta del proyecto].
2) Compilar el ejecutable con el archivo Makefile. Esto se hace con el comando make.
3) Ejecutar el programa con la siguiente sintaxis: ./OCVTest [operación a realizar] [ruta de la imagen a procesar].
La imagen resultante se encontrará en la carpeta del proyecto.
