# import tensorflow as tf
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt


# # make the model ready
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# model = Sequential([
#     base_model,
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(1, activation='sigmoid')  # binary 0 for brain 1 for no brain
# ])


# base_model.trainable = False
# model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# train_data = data_gen.flow_from_directory(
#     r'D:\data\train',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary'
# )

# val_data = data_gen.flow_from_directory(
#     r'D:\data\val',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary'
# )



# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=5
# )



# model.save('brain_classifier.h5')