from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
# load the model
model = VGG16()
# image Path
image_path ='.//mug.jpg'
# load an image from file
image = load_img(image_path, target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
print(yhat.shape)
# convert the probabilities to class labels
label = decode_predictions(yhat)

# retrieve the most likely result, Top 5 highest probability and print them
for i in range(5):
     print('%s (%.2f%%)' % (label[0][i][1], label[0][i][2]*100))
