# Banorte: Test tecnico
## Exploracion
Para esta prueba tecnica se utilizaron tres conjuntos de datos relacionados a la calificacion que cada usuario asigna a una pelicula. El primer conjunto contiene informacion de las calificaciones asignadas a cada pelicula vista por los usuarios y los otros conjuntos de datos tienen informacion del usuario como genero y edad y finalmente, informacion de la pelicula como titulo y genero. El primer paso de la exploracion fue revisar la existencia de datos faltantes y no fue el caso, por lo que posibles tecnicas de imputacion no fueron necesarias; despues se reviso que tampoco existieran registros repetidos para los datos.
Los datos son en su mayoria categoricos y en efecto solo aparecen las categorias reportadas en la metadata por lo que se empezo a construir desde este punto la matriz que serviria para un modelo supervisado. Lo primero fue separar la columna del genero de la pelicula en tantas columnas como generos tuviera, esto con la idea de que al modelo le puediera ser util la presencia o no de un genero en una pelicula.
Decidi no tomar en cuenta algunas features como Timestamp y codigo postal, esto por simplificar y porque los codigos postales son variables categoricas de alta dimension que pueden dar dificultades de ruido y computacionales al modelo.
## Modelo SVD y OHE
La idea intuitiva del abordaje SVD es explicar a los usuarios y peliculas en terminos de variables desconocidas (factores latentes), basandonos en como califican, y despues, para un usuario que no ha visto una pelicula, predecir el rating que le dara segun esas caracteristicas. 
Posteriormente pretendo usar este pronostico como feature de un modelo supervisado para incluir otros datos como edad, ocupacion, etc. Para finalizar esta parte se realizo el encoding de las variables categoricas y, como se menciono, el pronostico del modelo SVD se agrego como feature al dataframe. El notebook usado hasta ahora es exploration.ipynb.
# Modelo Xgboost y Shapley values
Una vez con el dataframe listo para ser usado como input del modelo, se hace la separacion train/test y entrenamos con los parametros estandar. Se abordo como un problema de regresion y no de clasificacion; aunque los ratings pudieran parecer categorias, no lo son, es una escala y por tanto una variable ordinal y en este sentido se utilizo la metrica MAE para medir su efectividad en el test set.
Los pronosticos se redondearon y asi se obtuvo el MAE. Los shapley values sirven para explicar las predicciones de algoritmos complejos pues proporciona una medida de que tanto contribuye cada feature al pronostico. El notebook de esta parte es model.ipynb. A continuacion vemos la importancia global de las características:

![Importancia](figures/imagen.png)

# Ideas adicionales
Algunas cosas que quedaron pendientes son:
- Tuneo de los hiperparametros para el modelo SVD y xgboost a traves de un optimizador bayesiano. En el codigo main.py viene una referencia de como hacerlo.
- Los shap values son muy costosos de calcular pues se necesita determinar todas las posibles combinaciones de las features, entonces queda pendiente usar una metodologia mas optima.
- Probar el modelo con las features mas importantes y ver si esta simplificacion se traduce en mejoria.

# Comentarios adicionales
- El test incluye requirements.txt para poderlo replicar en un entorno virtual.
- Seria necesario cambiar la ruta de os.chdir() para que lea las rutas relativas, en caso de ejecutar el codigo.
