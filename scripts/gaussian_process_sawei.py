import numpy as np
import GPy
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error

class GPSawei:
    def __init__(self, X, y, variables, measurement_variance=None, min_iterations=30, stop_threshold=0.005, window_size=10):
        """Initialize the Gaussian process with dynamic stopping and SAWEI acquisition.

        Arguments:
        ----------
        X -- (numpy array) input of initial measurements
        y -- (numpy array) corresponding outputs
        variables -- (list of strings) variable names
        measurement_variance -- (numpy array) optional variance for each measurement
        min_iterations -- (int) minimum iterations before checking stopping
        stop_threshold -- (float) gradient threshold to stop
        window_size -- (int) number of recent covariances to consider
        """
        assert len(variables) == X.shape[1], "Number of variables is not consistent with dataset"
        assert X.shape[0] == y.shape[0], "X and y dimensions are not consistent"

        print(
            "Gaussian Process initialization:\n"
            + " X = {}, y = {}, len(variables) = {}".format(
                X.shape, y.shape, len(variables)
            )
        )

        kernel = GPy.kern.Exponential(input_dim=len(variables), ARD=True)
        self.model = GPy.models.GPRegression(X, y, kernel)

        if measurement_variance is not None:
            self.model.likelihood.variance = measurement_variance

        self.model.optimize()
        self.model.constrain_positive(".*")

        self.mu = None
        self.cov = None
        self.min_iterations = min_iterations
        self.stop_threshold = stop_threshold
        self.window_size = window_size
        self.cov_history = []
        self.variance_history = []

    def predict(self, X):
        """Perform a prediction with the current state of self.model"""
        mu, cov = self.model.predict(X)
        self.mu = mu
        self.cov = cov
        return mu, cov

    def get_max_covariance(self):
        """Return the maximum covariance and its index"""
        idx = np.argmax(self.cov)
        return self.cov[idx], idx

    def get_max_gradient_covariance(self, X):
        """Return the index with highest gradient of uncertainty"""
        _, var_grad = self.model.predictive_gradients(X)
        grad_norm = np.linalg.norm(var_grad, axis=1)
        idx = np.argmax(grad_norm)
        return idx, grad_norm[idx]

    def update_Xy(self, X_new, y_new, measurement_variance=None):
        """Update the dataset with new measurement and repeat optimization"""
        self.model.set_XY(X=X_new, Y=y_new)
        if measurement_variance is not None:
            self.model.likelihood.variance = measurement_variance
        self.model.optimize()
        self.model.constrain_positive(".*")

        if self.cov is not None:
            self.cov_history.append(np.mean(self.cov))

    def compute_ei(self, X, y_best):
        """Compute the standard Expected Improvement (EI)"""
        mu, cov = self.predict(X)
        sigma = np.sqrt(cov)
        with np.errstate(divide='warn'):
            Z = (mu - y_best) / sigma
            ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def compute_sawei(self, X, y_best):
        """Compute Self-Adjusting Weighted Expected Improvement (SAWEI)"""
        mu, cov = self.predict(X)
        sigma = np.sqrt(cov)

        # Expected Improvement
        ei = self.compute_ei(X, y_best)

        # Normalize standard deviation
        avg_std = np.mean(sigma)
        normalized_std = sigma / (avg_std + 1e-8)
        w = 1 / (1 + np.exp(-normalized_std))  # sigmoid-based weight

        sawei = w * ei
        self.variance_history.append(avg_std)
        return sawei

    def get_next_point_sawei(self, X_candidates, y_best):
        """Select the next point using SAWEI acquisition"""
        sawei_values = self.compute_sawei(X_candidates, y_best)
        best_idx = np.argmax(sawei_values)
        return X_candidates[best_idx], sawei_values[best_idx]

    def check_stopping_condition(self, X, y_true, mae_threshold=0.05):
        """
        Check whether active learning should stop based on MAE and covariance stability,
        and return when the gradient became stable.
        """
        if len(self.cov_history) < self.min_iterations or self.mu is None:
            return False, None

        y_pred = np.exp(self.mu)
        mae = mean_absolute_error(y_true, y_pred)

        initial_cov = self.cov_history[0]
        norm_cov = np.array(self.cov_history) / initial_cov

        if len(norm_cov) < self.window_size:
            return False, None

        recent = norm_cov[-self.window_size:]
        gradient = np.gradient(recent)
        stable = np.all(np.abs(gradient) < self.stop_threshold)

        print(f"Check stopping: MAE = {mae:.5f}, Cov Gradient Stable = {stable}")
        gradient_index = len(self.cov_history) if stable else None

        if stable and mae <= mae_threshold:
            print(f"Stopping criteria met: MAE ≤ {mae_threshold} and covariance stabilized.")
            return True, gradient_index

        return False, gradient_index


