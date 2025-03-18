import numpy as np
import scipy.stats
import sklearn
from matplotlib import pyplot as plt
import tensorflow as tf


class Evaluation:
    """
    Evaluation of performance
    """

    def __init__(self, X_test, y_test):
        """
        :param X_test: test data with features
        :param y_test: test data with labels
        """
        self.X_test = X_test
        self.y_test = y_test
        self.frequency = np.linspace(0.6, 3.3, 222)

    def start_predictions(self, filename, model):
        """
        Starting training for predictions using either a saved model or default/new trained model
        :param filename: filename of saved model
        :param model: default/ new trained model
        :return:
        """

        if model is None:
            loaded_model = tf.keras.models.load_model(filename)
            predictions = loaded_model.predict([self.X_test], batch_size=1)
            return predictions

        elif filename is None:
            predictions = model.predict([self.X_test], batch_size=1)
            return predictions

    def evaluation_graphical(self, predictions):
        """
        Graphical evaluation showing predictions and true probabilities of heart rates
        :param predictions: predicted probabilities of heart rates
        :return:
        """
        # Create a figure and subplots
        fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(45, 25))

        # Iterate through the data and create subplots
        for i, ax in enumerate(axes.flat):
            if i < len(predictions):
                ax.plot(self.frequency, self.y_test[i*5][-1])
                ax.plot(self.frequency, predictions[i*5])
                ax.set_title(f"Graph {i*5 + 1}")
                ax.set_xlabel("Frequency in Hz")
                ax.set_ylabel("Probability")
                ax.legend(["Ground Truth", "Prediction"])
        # Adjust layout
        plt.tight_layout()
        # Show or save the figure
        plt.show()

    def evaluation_mae(self, predictions):
        """
        Evaluation of performance using weighted Mean Absolute Error (MAE)
        :param predictions: predicted probabilities of heart rates
        :return:
        """
        true_HR = np.zeros((len(self.y_test), 1))
        pred_HR = np.zeros((len(self.y_test), 1))
        for i in range(len(self.y_test)):
            max_idx = np.argmax(predictions[i])

            neigh_range = 5

            indices = []
            probabilities = []

            for j in range(max_idx - neigh_range, max_idx + neigh_range + 1):
                if 0 <= j < len(predictions[i]):
                    indices.append(j)
                    probabilities.append(predictions[i][j])

            weighted_max = np.average(indices, weights=probabilities)

            true_HR[i] = ((np.argmax(self.y_test[i][-1]) * (2.7 / 222)) + 0.6) * 60
            pred_HR[i] = ((weighted_max * (2.7 / 222)) + 0.6) * 60

        mae_bpm = sklearn.metrics.mean_absolute_error(true_HR, pred_HR)

        print("Mean Absolute Error (bpm): ", mae_bpm)

        return mae_bpm, true_HR, pred_HR

    def evaluation_are(self, true_HR, pred_HR):
        """
        Evaluation of the performance using the weighted mean of the absolute relative error (RAE)
        :param true_HR: true Heart rates
        :param pred_HR: predicted Heart rates
        :return:
        """
        absolute_errors = np.abs(true_HR - pred_HR)

        relative_errors = absolute_errors/true_HR

        average_are = 100 * sum(relative_errors) / len(relative_errors)

        print("Average Relative Absolute Error (%): ", float(average_are))

        return average_are

    def evaluation_bland_altman(self, true_HR, pred_HR):
        """
        Evaluation of the performance using the Bland-Altman-Plot
        :param true_HR: true Heart rates
        :param pred_HR: predicted Heart rates
        :return:
        """
        mean = np.mean([true_HR, pred_HR], axis=0)
        diff = true_HR - pred_HR
        mean_diff = np.mean(diff)
        sd_diff = np.std(diff)

        print(f"Mean difference: {mean_diff}, Standard deviation difference: {sd_diff}")

        plt.figure(figsize=(8, 6))
        plt.scatter(mean, diff, alpha=0.5)
        plt.axhline(y=mean_diff, color='red', linestyle='--', label='Mean Difference')
        plt.axhline(y=mean_diff + 1.96 * sd_diff, color='gray', linestyle='--',
                    label='Upper Limit of Agreement')
        plt.axhline(y=mean_diff - 1.96 * sd_diff, color='gray', linestyle='--',
                    label='Lower Limit of Agreement')
        plt.text(160, 40, fr"$\mu+1.96\sigma$={np.round(mean_diff+1.96*sd_diff, decimals=3)}", fontsize=10, color="black")
        plt.text(160, -40, fr"$\mu-1.96\sigma$={np.round(mean_diff-1.96*sd_diff, decimals=3)}", fontsize=10, color="black")
        plt.xlabel('Mean of True HR and Predicted HR [bpm]')
        plt.ylabel('Difference of True HR and Predicted HR [bpm]')
        plt.ylim(-60, 100)
        plt.title('Bland-Altman Plot')
        plt.legend()
        plt.show()

    def evaluate_corr(self, true_HR, pred_HR):
        """
        Evaluation of the performance using the Pearson Correlation Coefficient
        :param true_HR: true Heart rates
        :param pred_HR: predicted Heart rates
        :return:
        """
        true_HR = true_HR.ravel()
        pred_HR = pred_HR.ravel()
        corr_coeff, _ = scipy.stats.pearsonr(true_HR, pred_HR)

        slope, intercept, r_value, _, _ = scipy.stats.linregress(true_HR, pred_HR)
        lin_reg = slope * true_HR + intercept

        plt.figure()
        plt.scatter(true_HR, pred_HR, alpha=0.5)
        plt.plot(true_HR, lin_reg, color="red", label="Linear relationship")
        plt.text(80, 180, fr"$r={np.round(corr_coeff, 4)}$", fontsize=10, color="black")
        plt.title(f"Pearson Correlation")
        plt.xlabel("True HR [bpm]")
        plt.ylabel("Predicted HR [bpm]")
        plt.show()
