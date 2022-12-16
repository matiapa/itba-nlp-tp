Un análisis realizado sobre el dataset [Politifact](https://www.kaggle.com/datasets/rmisra/politifact-fact-check-dataset) para la materia de NLP.

El dataset contiene 21.152 afirmaciones en inglés de diversas fuentes, con un valor de veracidad asignado por el equipo de fact-cheking de Politifact.

Se implementó la clasificación del dataset utilizando cuatro estrategias:
- Estrategia 1 | Vectorización: Word2Vec | Clasificación: SVM

- Estrategia 2 | Vectorización: Word2Vec | Clasificación: KNN

- Estrategia 3 | Vectorización: Count vector | Clasificación: Random Forest

- Estrategia 4 | Vectorización: Count vector | Clasificación: Naive Bayes

Para cada estrategia se exploró el espacio de hiperparámetros para determinar la mejor configuración, y se realizaron las matrices de confusión.