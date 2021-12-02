# Dog Breed Classification with Deep Learning Methods

### Description rapide du projet

Ce projet est à la base d'une application déployée en ligne. Il permet la classification d'images de chiens en fonction de leur race. Des méthodes de transfer learning ont été utilisé pour obtenir l'algorithme de computer vision associée à cette application. Le modèle choisi est le modèle Inception V3. Ce choix a été fait pour plusieurs raisons : 
- les performances obtenues sont plutôt bonnes (80% d'accuracy) ; 
- son poids est léger compte tenu des performances observées.
Les données utilisées pour entraîner ce modèle sont contenues dans le dataset "Dogs Stanford" et disponible à cette adresse : [Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

Le détail de la démarche utilisé est présenter dans le notebook P06_01_notebook. Plusieurs tests ont été réalisés pour choisir le modèle à déployer. 

Une fois le modèle entraîné, une application Web sous Streamlit a été utilisé. Le choix de ce package a permis d'obtenir quelque chose de rapidement fonctionnel et de facilement déployable et partageable. Le code de cette application se trouve dans le script Python P06_02_programme. Les packages nécessaires se trouvent dans le fichier requirements.txt. L'application est hébergé grâce à l'outil Streamlit Share et se trouve à cette adresse : [API](https://share.streamlit.io/sylvariane/classification_image_deep_learning/main/P06_02_programme.py).

### Outils utilisés
- Jupyter Notebook/Google Colab Pro
- Python 3.8.5
- Numpy
- Matplotlib
- PIL
- OpenCV
- Tensorflow/Keras
- Streamlit
