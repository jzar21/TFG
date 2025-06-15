# Desarrollo de un sistema basado en deep learning para la estimación de la edad usando imagen radiológica de rodilla

<div>
  <a href="https://github.com/jzar21/TFG/blob/main/LICENSE">
    <img alt="Code License" src="https://img.shields.io/github/license/jzar21/TFG"/>
  </a>

<img src="https://img.shields.io/pypi/pyversions/torch"/>

</div>

## Descripción

Este repositorio alberga el código y los recursos correspondientes mi Trabajo Fin de Grado (TFG) de la Universidad de Granada. El objetivo principal del proyecto es desarrollar un modelo de aprendizaje profundo capaz de estimar la edad de un individuo a partir de imágenes radiológicas de la rodilla.

## Requisitos

Para ejecutar este proyecto, es necesario contar con las siguientes dependencias:

* Python 3.8 o superior
* Pytorch
* scikit-learn
* numpy
* matplotlib
* pandas
* opencv-python

Estas dependencias están especificadas en el archivo `requirements.txt`.

## Instalación

Para configurar el entorno de desarrollo, sigue estos pasos:

1. Clona este repositorio:

   ```bash
   git clone https://github.com/jzar21/TFG.git
   cd TFG
   ```

2. Crea y activa un entorno virtual (recomendado hacerlo mediante `conda`):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para entrenar el modelo, ejecuta el siguiente comando:

```bash
python main.py config.json
```

## Contribuciones

Las contribuciones al proyecto son bienvenidas. Si deseas colaborar, por favor sigue estos pasos:

1. Realiza un fork de este repositorio.
2. Crea una nueva rama para tu funcionalidad o corrección de errores.
3. Realiza tus cambios y asegúrate de que las pruebas pasen.
4. Envía un pull request detallando tus cambios.

## Licencia

Este proyecto está licenciado bajo la Licencia Pública General GNU v3.0. Para más detalles, consulta el archivo `LICENSE`.