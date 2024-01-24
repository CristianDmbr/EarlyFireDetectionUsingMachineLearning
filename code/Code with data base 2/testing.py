from keras.preprocessing import image
from keras.models import load_model
import numpy as np

cnn = load_model('Database/models/best_model.h5')

test_image = image.load_img('Database/neo_data/train/fire/WEB09412.jpg', target_size=(256, 256))
test_image = image.img_to_array(test_image)

# Rescale and resizing images for testing
test_image = test_image / 255.0

test_image = np.expand_dims(test_image, axis=0)

# To get prediction
prediction = cnn.predict(test_image)

print("Raw Prediction:", prediction)

# Check if the probability of the positive class (fire) is greater than or equal to 0.5
if prediction[0][0] >= 0.5:
    print("There is no fire")
else:
    print("There is fire")
