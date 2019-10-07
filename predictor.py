import cv2
from keras.models import load_model

def get_model():
    global model
    global modelVGG
    model = load_model('static/models/model.h5')
    modelVGG = load_model('static/models/modelVGG.h5')

def make_predictions(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("static/uploads/gray.jpg", gray)
    face_cascade = cv2.CascadeClassifier('static/models/haarcascade_frontalface_default.xml')
    img = cv2.imread("static/uploads/gray.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        for (x,y,w,h) in faces:
            face_clip = img[y:y+h, x:x+w]
            cv2.imwrite("static/uploads/gray.jpg", cv2.resize(face_clip, (350, 350)))
    else:
        cv2.imwrite("static/uploads/gray.jpg", cv2.resize(img, (350, 350)))
    read_image = cv2.imread("static/uploads/gray.jpg")
    read_image = read_image.reshape(1, read_image.shape[0], read_image.shape[1], read_image.shape[2])
    read_image_final = read_image/255.0
    VGG_Pred = modelVGG.predict(read_image_final)
    VGG_Pred = VGG_Pred.reshape(1, VGG_Pred.shape[1]*VGG_Pred.shape[2]*VGG_Pred.shape[3])
    top_pred = model.predict(VGG_Pred)
    return top_pred

def predict(dir):
    img = cv2.imread(dir)
    predictions = make_predictions(img)
    return predictions

if __name__=='__main__':
    get_model()
    predictions = predict("static/test/surprise.jpg")
    print(predictions[0])

