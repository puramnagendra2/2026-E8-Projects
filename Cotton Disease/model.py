import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

train_data_path = "train"
validation_data_path = "val"

training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

training_data = training_datagen.flow_from_directory(train_data_path, 
                                      target_size=(150, 150), 
                                      batch_size=32,
                                      class_mode='binary')  
 
print("Indices ",training_data.class_indices)

valid_datagen = ImageDataGenerator(rescale=1./255)
 
# this is a similar generator, for validation data
valid_data = valid_datagen.flow_from_directory(validation_data_path,
                                  target_size=(150,150),
                                  batch_size=32,
                                  class_mode='binary')

# Model Save Path
model_path = "saved_model.h5"
# Save only the best model
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Define Model
model = Sequential([
    Conv2D(32, kernel_size=3, input_shape=[150, 150, 3], activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(6, activation='softmax')  # Assuming 6 output classes
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train Model
history = model.fit(training_data, epochs=20, verbose=1, 
          validation_data=valid_data, callbacks=callbacks_list)
# Save Final Model
model.save(model_path)
print("Training Complete. Model Saved as 'pred_cotton.h5'")

# Testing
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load the test dataset
test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")