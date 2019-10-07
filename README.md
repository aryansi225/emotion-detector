# Emotion Detector
Detects emotion from an image of face

This is a Flask application that spits the probability of an image entered, being of following categories: Anger, Disgust, Fear, Happy, Neutral, Sad and Surprise. Additional it also displays the dominant emotion.

The model was created using keras and ipython notebook for the same is in the scripts folder.

Following are the steps followed in the notebook:
1. The data for human faces were taken from https://github.com/muxspace/facial_expressions repository.
2. The cleanup and preprocessing of human faces data was done and it is in Human_Face_Data_Cleanup ipython notebook.
3. The data for animated faces were taken from https://grail.cs.washington.edu/projects/deepexpr/ferg-db.html website.
4. The cleanup and preprocesing of animated faces data was done and it is in Animated_Face_Cleanup ipython notebook.
5. The main model building and training is done in Emotion_Detection ipython notebook.
6. First VGG-16 is used to get the bottleneck features of the grayscaled image.
7. Then CNN (Sequential -> Dense -> Dropout -> Dense -> Dense -> BatchNormalization -> Dense -> Dense)is built to do a multi class classification.
8. Finally the prediction using the model is done in Emotion_Prediction ipython notebook.

The models or pickeled objects are not in models folder since it would increase the size of repository, but it can be easily created by running the notebook.

LIVE DEMO HERE -> https://emotiondetectormachine.appspot.com/

I have deployed this in GCP and since in standard App Engine nothing can be written to disk except /tmp directory so in all places /tmp directory was used. Moreover for fecthing the image back to frontend the image was converted to Base64 and sent as the frontend was not able to fetch from /tmp directory. 

# Screenshots
![image](https://user-images.githubusercontent.com/16362957/66321070-849a5980-e90f-11e9-8e4b-98c41896ecb4.png)

![image](https://user-images.githubusercontent.com/16362957/66321167-b27f9e00-e90f-11e9-831e-5467b53783cd.png)

![image](https://user-images.githubusercontent.com/16362957/66321580-536e5900-e910-11e9-9cf1-3bc8e9099bf9.png)

# Dependencies
Flask, Tensorflow, Keras, OpenCV

# References
https://github.com/gauravtheP/Real-Time-Facial-Expression-Recognition
@inproceedings{aneja2016modeling,
  title={Modeling Stylized Character Expressions via Deep Learning},
  author={Aneja, Deepali and Colburn, Alex and Faigin, Gary and Shapiro, Linda and Mones, Barbara},
  booktitle={Asian Conference on Computer Vision},
  pages={136--153},
  year={2016},
  organization={Springer}
}
