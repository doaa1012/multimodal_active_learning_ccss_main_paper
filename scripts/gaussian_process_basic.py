import numpy as np
import GPy
from sklearn.metrics import mean_absolute_error

class GPBasic:
    def __init__(self, X, y, variables, measurement_variance=None, min_iterations=30, stop_threshold=0.005, window_size=10):
        """Initialize the Gaussian process with dynamic stopping

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

        # Check for consistency in dataset
        assert len(variables) == X.shape[1], "Number of variables is not consistent with dataset"
        assert X.shape[0] == y.shape[0], "X and y dimensions are not consistent"

        print(
            "Gaussian Process initialization:\n"
            + " X = {}, y = {}, len(variables) = {}".format(
                X.shape, y.shape, len(variables)
            )
        )

        # Use Exponential kernel (same as original)
        kernel = GPy.kern.Exponential(input_dim=len(variables), ARD=True)

        # Initialize model
        self.model = GPy.models.GPRegression(X, y, kernel)

        # Use heteroscedastic noise if provided
        if measurement_variance is not None:
            self.model.likelihood.variance = measurement_variance

        self.model.optimize()
        self.model.constrain_positive(".*")

        self.mu = None
        self.cov = None

        # Stopping parameters
        self.min_iterations = min_iterations
        self.stop_threshold = stop_threshold
        self.window_size = window_size
        self.cov_history = []

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

        # Update mean covariance history
        if self.cov is not None:
            self.cov_history.append(np.mean(self.cov))

    def check_stopping_condition(self, X, y_true, mae_threshold=0.005):
        """
        Check whether active learning should stop based on MAE and covariance stability,
        and return when the gradient became stable.
        """
        if len(self.cov_history) < self.min_iterations:
            return False, None

        if self.mu is None:
            return False, None

        # Convert log-scale predictions back
        y_pred = np.exp(self.mu)

        # Calculate Mean Absolute Error
        mae = mean_absolute_error(y_true, y_pred)

        # Normalize and check covariance stability
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


