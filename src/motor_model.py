import numpy as np
import random

class Motor:
    """
    A class to simulate a motor's state, including its degradation process,
    and apply an Extended Kalman Filter (EKF) to estimate its true state
    from noisy sensor measurements.
    """
    def __init__(self):
        # --- EKF Configuration for a 2D state: [vibration, temperature] ---
        self.Q = np.array([[1e-5, 0], 
                           [0, 1e-5]]) 
        self.R = np.array([[0.75**2, 0], 
                           [0, 0.25**2]]) 
        
        # --- Initial State ---
        self.true_state = np.array([[3.5], [85.5]]) # [true_vibration, true_temperature]
        self.x_hat = np.array([[3.5], [85.5]])
        self.P = np.array([[1.0, 0], 
                           [0, 1.0]])

        # --- PID-related State ---
        self.speed = 0.0

    def _f(self, x, voltage):
        """ 
        Non-linear state transition function.
        x is a 2x1 vector: [vibration, temperature]
        """
        vibration = x[0, 0]
        temperature = x[1, 0]

        # Vibration degradation model
        new_vibration = vibration + 0.05 + (vibration / 100.0)

        # Temperature model
        new_temperature = (temperature * 0.995) + (voltage * 0.01) + (vibration * 0.1)
        
        # --- FIXED: Add a physical limit to prevent runaway values ---
        if new_vibration > 100: new_vibration = 100
        if new_temperature > 200: new_temperature = 200

        return np.array([[new_vibration], [new_temperature]])

    def _h(self, x):
        """ Measurement function (linear, we directly measure the state). """
        return x

    def _F_jacobian(self, x, voltage):
        """ Jacobian of the state transition function f(x). """
        return np.array([[1.01, 0],
                           [0.1, 0.995]])

    def _H_jacobian(self, x):
        """ Jacobian of the measurement function h(x). """
        return np.array([[1.0, 0], 
                           [0, 1.0]])

    def _ekf_predict_update(self, x_hat, P, z, voltage):
        """ Performs one step of the 2D Extended Kalman Filter. """
        # Predict
        x_hat_minus = self._f(x_hat, voltage)
        F = self._F_jacobian(x_hat, voltage)
        P_minus = F @ P @ F.T + self.Q
        # Update
        H = self._H_jacobian(x_hat_minus)
        y_tilde = z - self._h(x_hat_minus)
        S = H @ P_minus @ H.T + self.R
        K = P_minus @ H.T @ np.linalg.inv(S)
        x_hat_new = x_hat_minus + K @ y_tilde
        P_new = (np.eye(2) - K @ H) @ P_minus
        
        return x_hat_new, P_new

    def update_state(self, voltage):
        """
        Simulates one time-step of the motor's operation based on input voltage
        and returns the new state.
        """
        self.true_state = self._f(self.true_state, voltage)
        
        vibration_noise = np.random.normal(0, 0.75)
        temperature_noise = np.random.normal(0, 0.25)
        noisy_measurement = self.true_state + np.array([[vibration_noise], [temperature_noise]])
        
        self.x_hat, self.P = self._ekf_predict_update(self.x_hat, self.P, noisy_measurement, voltage)
        
        self.speed = (self.speed * 0.9) + (voltage * 10) - (self.true_state[0, 0] / 5.0) + random.uniform(-5, 5)
        if self.speed < 0: self.speed = 0

        return {
            "true_vibration": self.true_state[0, 0],
            "true_temperature": self.true_state[1, 0],
            "noisy_vibration": noisy_measurement[0, 0],
            "noisy_temperature": noisy_measurement[1, 0],
            "ekf_vibration": self.x_hat[0, 0],
            "ekf_temperature": self.x_hat[1, 0],
            "speed": self.speed
        }
