"""
Author: Anton Altmeyer
Date: 25.06.2023

Sensor data fusion of PPG and ACC data for heart rate estimation using neural network
"""

import sklearn.model_selection
import tensorflow as tf
from Evaluation import Evaluation
from ModelGenerator import ModelGenerator
from Preprocessing import Preprocessing
import numpy as np
import data_loader


def custom_loss(y_true, y_pred):
    """
    Custom loss function
    :param y_true: true labels
    :param y_pred: predicted labels
    :return:
    """
    base_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    sigma = 3 / 60
    cost = (-1) * tf.math.log(base_loss * (tf.math.exp(-(y_true ** 2) / (2 * sigma ** 2)))
                              / (tf.reduce_max(tf.math.exp(-(y_true ** 2) / (2 * sigma ** 2)))))

    return cost


def main():
    """
    Main
    :return:
    """
    preprocessor = Preprocessing(sample_frequency=50)
    X_BAMI1, y_BAMI1, intensity_BAMI1 = preprocessor.processing_BAMI1()
    X_BAMI2, y_BAMI2, intensity_BAMI2 = preprocessor.processing_BAMI2()
    X_ISPC, y_ISPC, intensity_ISPC = preprocessor.preprocessing_ISPC()

    X_BAMI, y_BAMI, intensity_BAMI = \
        np.vstack((X_BAMI1, X_BAMI2)), np.vstack((y_BAMI1, y_BAMI2)), np.vstack((intensity_BAMI1, intensity_BAMI2))

    X_train_BAMI, X_test_BAMI, y_train_BAMI, y_test_BAMI, intensity_train_BAMI, intensity_test_BAMI = \
        sklearn.model_selection.train_test_split(X_BAMI, y_BAMI, intensity_BAMI, test_size=0.3, shuffle=False)

    X_BAMI1_ISPC, y_BAMI1_ISPC, intensity_BAMI1_ISPC = \
        np.vstack((X_BAMI1, X_ISPC)), np.vstack((y_BAMI1, y_ISPC)), np.vstack((intensity_BAMI1, intensity_ISPC))

    # create the model generator for the neural network
    model_generator = ModelGenerator(time_steps=6, input_height=2, input_width=222, input_channels=1)
    model = model_generator.generate_model()
    model.summary()

    # model.fit([X_BAMI1_ISPC, intensity_BAMI1_ISPC], y_BAMI1_ISPC[:, -1, :], epochs=40, batch_size=1)
    # model.save("model\model_epochs40_lossCategoricalCrossentropy")

    # Test data evaluation
    evaluator = Evaluation(X_test=[X_BAMI2, intensity_BAMI2], y_test=y_BAMI2)
    predictions = evaluator.start_predictions(
        filename="model\model_epochs40_lossCategoricalCrossentropy", model=None)
    evaluator.evaluation_graphical(predictions=predictions)
    _, true_HR, pred_HR = evaluator.evaluation_mae(predictions=predictions)
    _ = evaluator.evaluation_are(true_HR=true_HR, pred_HR=pred_HR)
    evaluator.evaluation_bland_altman(true_HR=true_HR, pred_HR=pred_HR)
    evaluator.evaluate_corr(true_HR=true_HR, pred_HR=pred_HR)

    # Train data evaluation
    evaluator = Evaluation(X_test=[X_BAMI1_ISPC, intensity_BAMI1_ISPC], y_test=y_BAMI1_ISPC)
    predictions = evaluator.start_predictions(
        filename="model\model_epochs40_lossCategoricalCrossentropy", model=None)
    evaluator.evaluation_graphical(predictions=predictions)
    _, true_HR, pred_HR = evaluator.evaluation_mae(predictions=predictions)
    _ = evaluator.evaluation_are(true_HR=true_HR, pred_HR=pred_HR)
    evaluator.evaluation_bland_altman(true_HR=true_HR, pred_HR=pred_HR)
    evaluator.evaluate_corr(true_HR=true_HR, pred_HR=pred_HR)

    evaluator = Evaluation(X_test=[X_ISPC, intensity_ISPC], y_test=y_ISPC)
    predictions = evaluator.start_predictions(
        filename="model\model_epochs40_lossCategoricalCrossentropy_trainsetBAMItrain_testsetBAMItest", model=None)
    evaluator.evaluation_graphical(predictions=predictions)
    _, true_HR, pred_HR = evaluator.evaluation_mae(predictions=predictions)
    _ = evaluator.evaluation_are(true_HR=true_HR, pred_HR=pred_HR)
    evaluator.evaluation_bland_altman(true_HR=true_HR, pred_HR=pred_HR)
    evaluator.evaluate_corr(true_HR=true_HR, pred_HR=pred_HR)

    evaluator = Evaluation(X_test=[X_test_BAMI, intensity_test_BAMI], y_test=y_test_BAMI)
    predictions = evaluator.start_predictions(
        filename="model\model_epochs40_lossCategoricalCrossentropy_trainsetBAMItrain_testsetBAMItest", model=None)
    evaluator.evaluation_graphical(predictions=predictions)
    _, true_HR, pred_HR = evaluator.evaluation_mae(predictions=predictions)
    _ = evaluator.evaluation_are(true_HR=true_HR, pred_HR=pred_HR)
    evaluator.evaluation_bland_altman(true_HR=true_HR, pred_HR=pred_HR)
    evaluator.evaluate_corr(true_HR=true_HR, pred_HR=pred_HR)

    evaluator = Evaluation(X_test=[X_train_BAMI, intensity_train_BAMI], y_test=y_train_BAMI)
    predictions = evaluator.start_predictions(
        filename="model\model_epochs40_lossCategoricalCrossentropy_trainsetBAMItrain_testsetBAMItest", model=None)
    evaluator.evaluation_graphical(predictions=predictions)
    _, true_HR, pred_HR = evaluator.evaluation_mae(predictions=predictions)
    _ = evaluator.evaluation_are(true_HR=true_HR, pred_HR=pred_HR)
    evaluator.evaluation_bland_altman(true_HR=true_HR, pred_HR=pred_HR)
    evaluator.evaluate_corr(true_HR=true_HR, pred_HR=pred_HR)

    ##########################################cross-fold-validation#####################################################

    num_folds = 4
    kf = sklearn.model_selection.KFold(n_splits=num_folds, shuffle=True, random_state=42)

    model_generator = ModelGenerator(time_steps=6, input_height=2, input_width=222, input_channels=1)

    epoch_range = [30, 35, 40, 45, 50, 55]

    fold_metrics = []

    for num_epochs in epoch_range:

        epoch_metrics = []

        print(f"Epochs: {num_epochs}")

        for train_idx, val_idx in kf.split(X_BAMI1):
            X_train_BAMI1, X_val_BAMI1 = X_BAMI1[train_idx], X_BAMI1[val_idx]
            y_train_BAMI1, y_val_BAMI1 = y_BAMI1[train_idx], y_BAMI1[val_idx]
            intensity_train_BAMI1, intensity_val_BAMI1 = intensity_BAMI1[train_idx], intensity_BAMI1[val_idx]

            model_kf = model_generator.generate_model()

            model_kf.fit([X_train_BAMI1, intensity_train_BAMI1], y_train_BAMI1[:, -1, :], epochs=num_epochs,
                         batch_size=1)

            evaluator_kf = Evaluation(X_test=[X_val_BAMI1, intensity_val_BAMI1], y_test=y_val_BAMI1)
            predictions_kf = evaluator_kf.start_predictions(filename=None, model=model_kf)
            mae_kf = evaluator_kf.evaluation_mae(predictions=predictions_kf)

            epoch_metrics.append(mae_kf)

        mean_mae = np.mean(epoch_metrics)

        fold_metrics.append((num_epochs, mean_mae))

    for num_epochs, mean_mae in fold_metrics:
        print(f"Number of epochs: {num_epochs}, Mean MAE: {mean_mae}")


if __name__ == '__main__':
    main()
