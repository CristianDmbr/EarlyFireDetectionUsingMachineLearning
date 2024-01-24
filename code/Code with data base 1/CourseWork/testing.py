from keras.preprocessing import image
from keras.models import load_model
import numpy as np

cnn = load_model('Database/models/best_model.h5')

test_image = image.load_img('CourseWork/Database/Training Data/Fire/fire-98.1979223526.png', target_size=(256, 256))
test_image = image.img_to_array(test_image)

# Rescale and resizing images for testing

test_image = test_image / 255

test_image = np.expand_dims(test_image, axis=0)

# To get prediction
prediction = cnn.predict(test_image)

# Since you're using binary classification (fire or smoke), you can use the predicted value directly
if prediction >= 0.5:
    print("It is smoke")
else:
    print("It is fire")
