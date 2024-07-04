from glob import glob
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint
from keras.src.layers import GlobalAveragePooling2D, Dense
from sklearn.datasets import load_files
from tensorflow.python.keras.utils import np_utils
from keras.preprocessing import image
import matplotlib.pyplot as plt
from random import sample

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def extract_Resnet50(tensor):
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def Resnet50_predict_breed(img_path):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = model_Resnet50.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

train_files, train_targets = load_dataset('./input/train')
valid_files, valid_targets = load_dataset('./input/valid')
test_files, test_targets = load_dataset('./input/test')

dog_names = [item[18:-1] for item in sorted(glob("./input/train/*/"))]

print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.' % len(test_files))

bottleneck_features = np.load('./data/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']

model_Resnet50 = Sequential()
model_Resnet50.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
model_Resnet50.add(Dense(133, activation='softmax'))

model_Resnet50.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50_model.keras', verbose=1,
                               save_best_only=True)
history = model_Resnet50.fit(train_Resnet50, train_targets,
                   validation_data=(valid_Resnet50, valid_targets),
                   epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

model_Resnet50.load_weights('saved_models/weights.best.Resnet50_model.keras')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# get index of predicted dog breed for each image in test set
Resnet50_predictions = [np.argmax(model_Resnet50.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

# report test accuracy
test_accuracy = 100 * np.sum(np.array(Resnet50_predictions) == np.argmax(test_targets, axis=1)) / len(Resnet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

# Predict and plot six different images
random_indices = sample(range(len(test_files)), 6)
test_images = [test_files[i] for i in random_indices]
actual_breeds = [dog_names[np.argmax(test_targets[i])] for i in random_indices]
predicted_breeds = [Resnet50_predict_breed(img) for img in test_images]

plt.figure(figsize=(20, 10))

for i in range(6):
    img = image.load_img(test_images[i], target_size=(224, 224))
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Actual: {actual_breeds[i]}\nPredicted: {predicted_breeds[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()