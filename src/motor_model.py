import numpy as np
import random

class Motor:
    """
    A class to simulate a motor's state, now including both vibration and temperature.
    It applies an upgraded Extended Kalman Filter (EKF) to fuse both sensor
    measurements into a single, robust state estimate.
    """
    def __init__(self):
        # --- EKF Configuration for a 2D state: [vibration, temperature] ---
        # Process noise covariance (Q): How much we trust our process model.
        # We assume vibration and temperature degradation are mostly independent.
        self.Q = np.array([[1e-5, 0], 
                           [0, 1e-5]]) 
        
        # Measurement noise covariance (R): How much we trust the sensors.
        # We model the vibration sensor as slightly noisier than the temperature sensor.
        self.R = np.array([[0.75**2, 0], 
                           [0, 0.25**2]]) 
        
        # --- Initial State ---
        self.true_state = np.array([[3.5], [85.5]]) # [true_vibration, true_temperature]
        
        # Initial state estimate for EKF
        self.x_hat = np.array([[3.5], [85.5]])
        
        # Initial error covariance for EKF
        self.P = np.array([[1.0, 0], 
                           [0, 1.0]])

        # --- PID-related State ---
        self.speed = 0.0 # Current motor speed in RPM

    def _f(self, x, voltage):
        """ 
        Non-linear state transition function.
        x is a 2x1 vector: [vibration, temperature]
        """
        vibration = x[0, 0]
        temperature = x[1, 0]

        # Vibration degradation model (same as before)
        new_vibration = vibration + 0.05 + (vibration / 100.0)

        # Temperature model: 
        # - It slowly cools towards ambient (e.g., 25C), represented by (temperature * 0.995).
        # - It heats up based on applied voltage (work done).
        # - It heats up significantly more as vibration (inefficiency) increases.
        new_temperature = (temperature * 0.995) + (voltage * 0.01) + (vibration * 0.1)

        return np.array([[new_vibration], [new_temperature]])

    def _h(self, x):
        """ Measurement function (linear, we directly measure the state). """
        return x

    def _F_jacobian(self, x, voltage):
        """ Jacobian of the state transition function f(x). """
        # Partial derivatives of _f with respect to vibration and temperature
        # d(new_vibration)/d(vibration) = 1.01
        # d(new_vibration)/d(temperature) = 0
        # d(new_temperature)/d(vibration) = 0.1
        # d(new_temperature)/d(temperature) = 0.995
        return np.array([[1.01, 0],
                           [0.1, 0.995]])

    def _H_jacobian(self, x):
        """ Jacobian of the measurement function h(x). """
        # H is the identity matrix because our measurement is linear
        return np.array([[1.0, 0], 
                           [0, 1.0]])

    def _ekf_predict_update(self, x_hat, P, z, voltage):
        """ Performs one step of the 2D Extended Kalman Filter. """
        # --- Predict Step ---
        x_hat_minus = self._f(x_hat, voltage)
        F = self._F_jacobian(x_hat, voltage)
        P_minus = F @ P @ F.T + self.Q

        # --- Update Step ---
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
        # 1. Simulate the true process
        self.true_state = self._f(self.true_state, voltage)
        
        # 2. Simulate noisy measurements
        vibration_noise = np.random.normal(0, 0.75)
        temperature_noise = np.random.normal(0, 0.25)
        noisy_measurement = self.true_state + np.array([[vibration_noise], [temperature_noise]])
        
        # 3. Use EKF to get a clean estimate of the state
        self.x_hat, self.P = self._ekf_predict_update(self.x_hat, self.P, noisy_measurement, voltage)
        
        # 4. Simulate motor speed
        self.speed = (self.speed * 0.9) + (voltage * 10) - (self.true_state[0, 0] / 5.0) + random.uniform(-5, 5)
        if self.speed < 0: self.speed = 0

        # 5. Return all relevant data points
        return {
            "true_vibration": self.true_state[0, 0],
            "true_temperature": self.true_state[1, 0],
            "noisy_vibration": noisy_measurement[0, 0],
            "noisy_temperature": noisy_measurement[1, 0],
            "ekf_vibration": self.x_hat[0, 0],
            "ekf_temperature": self.x_hat[1, 0],
            "speed": self.speed
        }
