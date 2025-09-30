# CAXX Radar Processing Tools

Herramientas en Python para el procesamiento de datos crudos del radar meteorológico **CAXX**, ubicado en el cerro Paraguillas (4.450 msnm), al norte del Parque Nacional Cajas y cerca de Cuenca, Ecuador.

El radar CAXX forma parte de la red **RadarNet-Sur**, instalada en los Andes tropicales para el monitoreo de precipitaciones en tiempo real.

## Objetivo del proyecto

Este repositorio implementa un pipeline para:

-   Leer archivos crudos `.azi` generados por el radar (dBuZ y dBZ).
-   Convertir datos polares (azimut, rango) a coordenadas UTM.
-   Interpolar los datos a una grilla regular (GeoTIFF / NetCDF).
-   Generar productos de reflectividad listos para investigación y aplicaciones hidrológicas.

## Estructura del proyecto

caxx-radar/
├─ data/ \# Datos crudos e intermedios
├─ products/ \# Figuras, logs y productos finales
├─ src/ \# Código fuente (lectura, georef, QPE)
├─ scripts/ \# Scripts de procesamiento
├─ config/ \# Parámetros (EPSG, radar, resolución, etc.)
├─ notebooks/ \# Exploración rápida
├─ tests/ \# Pruebas unitarias
├─ requirements.txt \# Dependencias de Python
└─ LICENSE \# Aviso de uso restringido

## Licencia

Este repositorio es de uso exclusivo de ETAPA EP.\
Queda prohibida la copia, distribución o uso sin autorización expresa.