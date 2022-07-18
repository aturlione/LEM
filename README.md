
# Funcionamiento

## Hydrobid

El modelo de hydrobid devuelve los resultados de una ejecución a partir de los inputs de usuario.

Cuando comienza la ejecución, se crea una carpeta temporal con el nombre de la ejecución, todos los resultados se crean en dicha carpeta, al finalizar la ejecución se borra la carpeta temporal.

Los resulados se devuelven en formato JSON, se envían todos los resultados.

Para cada carpeta de ejecución se copian los archivos necesarios, que son los siguientes:

    -   El fichero geojson con las cuencas.
    -   El fichero HydroBID-2.3.jar, es el ejecutable del modelo.
    -   El fichero Hydrobid.sqlite, la base de datos que necesita el ejecutable.
    -   El fichero settings.txt, configuración para la ejecución.

Los datos climáticos se almacenan en la tabla climate_data dentro del fichero Hydrobid.sqlite. Son medias diarias de precipitación y temperatura, la precipitación se almacena en cm y la temperatura en ºC.

## Regional stats

Método que devuelve las estadísticas de una variable de un producto para una región.
Método que se ha utilizado para la configuración y ejecución de los procesos.

Devuelve un JSON con las estadísticas de la variable para la región.