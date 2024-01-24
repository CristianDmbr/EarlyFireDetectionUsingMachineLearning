# Database from : https://github.com/gaiasd/DFireDataset

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Generate testing data
batch_size = 16

training_datagenerator = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.1
)

# 90% of the database goes to train
train = training_datagenerator.flow_from_directory(
    'Database/neo_data/train',
    target_size=(256, 256),
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size,
    subset='training'
)

# 10% of the database goes to validation
validation = training_datagenerator.flow_from_directory(
    'Database/neo_data/train',
    target_size=(256, 256),
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size,
    subset='validation'
)

# Initialising CNN
cnn = tf.keras.models.Sequential()

# Add first layer to CNN
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[256, 256, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))

# Add second layer
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))

# Add third layer
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Fully connected layer and add the amount of neurons
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output layer. 1 neuron because the output is either fire or smoke
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Train CNN model
checkpoint = tf.keras.callbacks.ModelCheckpoint('Database/models/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
callbacks = [checkpoint]

# Compile and train
cnn.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name='auc')])

# Training loop
for epoch in range(10):
    cnn.fit(train, validation_data=validation, steps_per_epoch=train.samples // batch_size,
            validation_steps=validation.samples // batch_size, callbacks=callbacks)

    # Evaluate on validation set
    evaluation_metrics = cnn.evaluate(validation)
    accuracy = evaluation_metrics[1]
    precision = evaluation_metrics[2]
    recall = evaluation_metrics[3]
    f1 = 2 * (precision * recall) / (precision + recall)

    # Print or log the metrics
    print(f"Epoch {epoch + 1} - Validation Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Save the final model
cnn.save('Database/models/final_model.h5')