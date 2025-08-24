import numpy as np

class Motor:
    """
    A class to simulate a motor's state, including its degradation process,
    and apply an Extended Kalman Filter (EKF) to estimate its true state
    from noisy sensor measurements.
    """
    def __init__(self):
        # --- EKF Configuration ---
        # State is a single value: [vibration]
        # Process noise covariance (Q): How much we trust our process model.
        # A small value means we trust our model a lot.
        self.Q = np.array([[1e-5]]) 
        # Measurement noise covariance (R): How much we trust the sensor measurement.
        # A larger value means the sensor is noisy.
        self.R = np.array([[0.75**2]]) 
        
        # --- Initial State ---
        self.true_vibration_state = 3.5
        # Initial state estimate for EKF
        self.x_hat = np.array([[3.5]])
        # Initial error covariance for EKF
        self.P = np.array([[1.0]])

    def _f(self, x):
        """ Non-linear state transition function. """
        degradation_rate = 0.05 + (x[0, 0] / 100.0)
        return np.array([[x[0, 0] + degradation_rate]])

    def _h(self, x):
        """ Measurement function (linear in this case). """
        return x

    def _F_jacobian(self, x):
        """ Jacobian of the state transition function f(x). """
        # Derivative of f(x) with respect to x is 1 + 1/100
        return np.array([[1.01]])

    def _H_jacobian(self, x):
        """ Jacobian of the measurement function h(x). """
        # Derivative of h(x) with respect to x is 1
        return np.array([[1.0]])

    def _ekf_predict_update(self, x_hat, P, z):
        """ Performs one step of the Extended Kalman Filter. """
        # --- Predict Step ---
        x_hat_minus = self._f(x_hat)
        F = self._F_jacobian(x_hat)
        P_minus = F @ P @ F.T + self.Q

        # --- Update Step ---
        H = self._H_jacobian(x_hat_minus)
        y_tilde = z - self._h(x_hat_minus)
        S = H @ P_minus @ H.T + self.R
        K = P_minus @ H.T @ np.linalg.inv(S)
        x_hat_new = x_hat_minus + K @ y_tilde
        P_new = (np.eye(1) - K @ H) @ P_minus
        
        return x_hat_new, P_new

    def update_state(self):
        """
        Simulates one time-step of the motor's operation and returns the new state.
        """
        # 1. Simulate the true process and noisy measurement
        degradation_rate = 0.05 + (self.true_vibration_state / 100.0)
        self.true_vibration_state += degradation_rate
        noise = np.random.normal(0, 0.75)
        noisy_vibration_measurement = self.true_vibration_state + noise
        
        # 2. Use EKF to get a clean estimate of the state
        z = np.array([[noisy_vibration_measurement]])
        self.x_hat, self.P = self._ekf_predict_update(self.x_hat, self.P, z)
        ekf_estimated_vibration = self.x_hat[0, 0]

        # 3. Return all relevant data points
        return {
            "true_vibration": self.true_vibration_state,
            "noisy_vibration": noisy_vibration_measurement,
            "ekf_vibration": ekf_estimated_vibration
        }
