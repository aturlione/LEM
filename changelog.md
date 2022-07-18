## v1.6.8

-   Se descuentan los caudales de subcuencas con embalses.

## v1.6.6

-   Corregido cálculo de demandas.

## v1.6.4

-   Cambiado el campo que se devuelve en hydrobid.

## v1.6.3

-   Añadido el resultado de hydrobid total.

## v1.6.2

-   Corregido cálculo de obtención de demandas.

## v1.6.1

-   Las unidades ya están en m3/day en base de datos, no se transforman.
-   No se ejecuta hydrobid para las cuencas aguas arriba.
-   La consulta custom devuelve el mismo formato que la consulta de cálculo.

## v1.6.0

-   Se ha agregado un método para usar parámetros customs.
-   Se ha agregado un método que cambia las unidades del output.
-   Se han agregado las aportaciones de agua en el output.
-   Se han modificado los métodos anteriores para que en el output no aparezcan las estaciones que quedan fuera del rango
    temporal que se utiliza al hacer la consula, por ejemplo si la consulta va desde septiembre del 2011 hasta enero del 2012,
    esta consuylta en el emisferio sur sólo abarca primavera y verano. Esto es para evitar que aparezcan valores negativos en el
    resultado final.
-   Ahora mismo todo está calculado considerando caudales en m3/day

## v1.5.12

-   Connection settings fixed.

## v1.5.11

-   Bug fixed calculating demands.

## v1.5.10

-   Updated units to m3/day.

## v1.5.9

-   Fixed units in result.

## v1.5.8

-   Changed average to total.

## v1.5.7

-   Removing custom params.
-   Fixed calculations.

## v1.5.6

-   Bugs fixed.
-   Custom demands.

## v1.5.5

-   Bugs fixed.

## v1.5.3

-   Updated methods in calculate volumens.

## v1.5.2

-   Updated version number.

## v1.5.1

-   Updated requirements.txt to python 3.6.

## v1.5.0

-   Clean parameters not used.
-   Hydrobid with climate change.
-   Added Calculate volumes.


## v1.4.2

-   Removed unused code.

## v1.4.0

-   Get categorized values.

## v1.3.0

-   Get climate data from Api SATD-Katari.

## v1.2.0

-   Removed catchments geojson from hydrobid files.
-   Added url geojson catchment input.
-   Updated calibrated parameters in hydrobid sqlite database. 

## v1.1.9

-   Updated summary result.

## v1.1.8

-   Updated conf.

## v1.1.7

-   Changed in temp folder.

## v1.1.6

-   Updated readme.

## v1.1.5

-   Updated results in hydrobid.

## v1.1.4

-   Extent and resolution added in inputs.

## v1.1.3

-   Moved hydrobid files.

## v1.1.2

-   Reorder hydrobid files.

## v1.1.1

-   Updated manifest.

## v1.1.0

-   Implementado modelo hydrobid.

## v1.0.0

>>>>>>> 03705a9bb640d2f62c38ffa94f9192656259f0b9
-   Permite devolver las estadísticas de una variable de un producto para una región.