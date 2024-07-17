from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input as prep_res
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as prep_vgg
from data_augmentation import get_path, data_gen_base, data_gen_sota
from sklearn.metrics import fbeta_score, roc_auc_score


def build_base_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.05))

    model.add(Dense(1, activation='sigmoid'))

    return model


def build_resnet_model():
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    resnet_model = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-1].output)

    x = resnet_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model_resn = Model(inputs=resnet_model.input, outputs=predictions)

    return model_resn


def build_vgg_model():
    vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    vgg_model = Model(inputs=vgg_model.input, outputs=vgg_model.layers[-3].output)

    vgg_output = vgg_model.output

    x = GlobalAveragePooling2D()(vgg_output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model_vgg = Model(inputs=vgg_model.input, outputs=predictions)

    return model_vgg


def train_and_save_model(model, train_gen, valid_data, epochs, model_path):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3,
                                                min_lr=0.000001)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=train_gen, epochs=epochs, validation_data=valid_data, callbacks=[learning_rate_reduction])
    model.save(model_path)


def metrics(model, unbalanced_data, balanced_data):
    y_preds_unbalanced = [1 if prob >= 0.5 else 0 for prob in model.predict(unbalanced_data)]
    y_preds_balanced = [1 if prob >= 0.5 else 0 for prob in model.predict(balanced_data)]

    y_true_unbalanced = unbalanced_data.classes
    y_true_balanced = balanced_data.classes

    f2_score_unbalanced = fbeta_score(y_true_unbalanced, y_preds_unbalanced, beta=2)
    f2_score_balanced = fbeta_score(y_true_balanced, y_preds_balanced, beta=2)

    roc_auc_unbalanced = roc_auc_score(y_true_unbalanced, y_preds_unbalanced)
    roc_auc_balanced = roc_auc_score(y_true_balanced, y_preds_balanced)

    return f2_score_unbalanced, f2_score_balanced, roc_auc_unbalanced, roc_auc_balanced


if __name__ == '__main__':
    train_dir, validation_dir, unbalanced_test_dir, balanced_test_dir = get_path()
    train_data_base, val_data_base, unbalanced_data_base, balanced_data_base = data_gen_base(train_dir,
                                                                                             validation_dir,
                                                                                             unbalanced_test_dir,
                                                                                             balanced_test_dir)
    train_data_res, val_data_res, unbalanced_data_res, balanced_data_res = data_gen_sota(train_dir,
                                                                                         validation_dir,
                                                                                         unbalanced_test_dir,
                                                                                         balanced_test_dir,
                                                                                         prep_res,
                                                                                         (128, 128))
    train_data_vgg, val_data_vgg, unbalanced_data_vgg, balanced_data_vgg = data_gen_sota(train_dir,
                                                                                         validation_dir,
                                                                                         unbalanced_test_dir,
                                                                                         balanced_test_dir,
                                                                                         prep_vgg,
                                                                                         (128, 128))

    base_model = build_base_model()
    train_and_save_model(base_model, train_data_base, val_data_base, epochs=15, model_path='base_model.h5')
    base_metrics = metrics(base_model, unbalanced_data_base, balanced_data_base)

    resnet_model = build_resnet_model()
    train_and_save_model(resnet_model, train_data_res, val_data_res, epochs=15, model_path='resnet_model.h5')
    resnet_metrics = metrics(resnet_model, unbalanced_data_res, balanced_data_res)

    vgg_model = build_vgg_model()
    train_and_save_model(vgg_model, train_data_vgg, val_data_vgg, epochs=15, model_path='vgg_model.h5')
    vgg_metrics = metrics(vgg_model, unbalanced_data_vgg, balanced_data_vgg)
