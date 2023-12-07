import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def plot_history(history,metric):
    plt.plot(history.history[metric])
    plt.plot(history.history["val_"+metric],"")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, "val_"+metric])
    plt.show()


train_dir = r"C:\Users\Sebastiao\Desktop\Projetos\projeto-ia-surface-defect\data\NEU Metal Surface Defects Data\train"
test_dir = r"C:\Users\Sebastiao\Desktop\Projetos\projeto-ia-surface-defect\data\NEU Metal Surface Defects Data\test"
valid_dir = r"C:\Users\Sebastiao\Desktop\Projetos\projeto-ia-surface-defect\data\NEU Metal Surface Defects Data\valid"

## Data preprocessing before modeling
train_datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

#get the images from train datagen
train_generator = train_datagen.flow_from_directory(train_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=True)
valid_generator = test_datagen.flow_from_directory(valid_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False)

test_generator = test_datagen.flow_from_directory(test_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False)

class_names = train_generator.class_indices
class_names = list(class_names.keys())

model1 = Sequential([ Conv2D(32, (2, 2), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(6 ,activation='softmax')])

model1.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

history = model1.fit(train_generator,
                    epochs=20,
                    batch_size=32,
                    validation_data=valid_generator)

model1.save("meu_modelo.h5")

plot_history(history,"accuracy")
plot_history(history,"loss")

result = model1.evaluate(test_generator)
print("Test loss, Test accuracy : ", result)

