from keras.src.legacy.preprocessing.image import ImageDataGenerator
from pathlib import Path


def get_path():
    train_dir = Path('dataset/train_another')
    validation_dir = Path('dataset/validation_another')
    unbalanced_test_dir = Path('dataset/test_another')
    balanced_test_dir = Path('dataset/test')
    return train_dir, validation_dir, unbalanced_test_dir, balanced_test_dir


def data_gen_base(train_dir, validation_dir, unbalanced_test_dir, balanced_test_dir):
    train_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rescale=1 / 255.0,
                                   brightness_range=[0.2, 1.2])
    validation_gen = ImageDataGenerator(rotation_range=10,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        rescale=1 / 255.0,
                                        brightness_range=(0.2, 1.2))

    test_unbalanced_gen = ImageDataGenerator()
    test_balanced_gen = ImageDataGenerator()

    train_data = train_gen.flow_from_directory(
        directory=train_dir,
        target_size=(128, 128),
        class_mode='binary',
        color_mode='rgb',
        shuffle=True,
        batch_size=100)

    val_data = validation_gen.flow_from_directory(
        directory=validation_dir,
        target_size=(128, 128),
        class_mode='binary',
        color_mode='rgb',
        shuffle=True,
        batch_size=100)

    unbalanced_data = test_unbalanced_gen.flow_from_directory(directory=unbalanced_test_dir,
                                                              target_size=(128, 128),
                                                              class_mode='binary',
                                                              shuffle=False,
                                                              color_mode='rgb',
                                                              batch_size=100)

    balanced_data = test_balanced_gen.flow_from_directory(directory=balanced_test_dir,
                                                          target_size=(128, 128),
                                                          class_mode='binary',
                                                          color_mode='rgb',
                                                          shuffle=False,
                                                          batch_size=100)

    return train_data, val_data, unbalanced_data, balanced_data


def data_gen_sota(train_dir, validation_dir, unbalanced_test_dir, balanced_test_dir, preprocess_input, target_size):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=10,
                                 width_shift_range=0.2,
                                 height_shift_range=0.3,
                                 zoom_range=0.4,
                                 horizontal_flip=True,
                                 brightness_range=[0.3, 1.1])

    test_unbalanced_gen_2 = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_balanced_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = datagen.flow_from_directory(directory=train_dir,
                                             target_size=target_size,
                                             class_mode='binary',
                                             batch_size=16)

    validation_data = datagen.flow_from_directory(directory=validation_dir,
                                                  target_size=target_size,
                                                  class_mode='binary',
                                                  batch_size=16)

    unbalanced_data = test_unbalanced_gen_2.flow_from_directory(directory=unbalanced_test_dir,
                                                                target_size=target_size,
                                                                class_mode='binary',
                                                                batch_size=16)

    balanced_data = test_balanced_gen.flow_from_directory(directory=balanced_test_dir,
                                                          target_size=target_size,
                                                          class_mode='binary',
                                                          batch_size=16)
    return train_data, validation_data, unbalanced_data, balanced_data
