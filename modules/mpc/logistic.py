# modules/mpc/logistic.py

from mpyc.runtime import mpc
from utils.constant import DEFAULT_EPOCHS, DEFAULT_LR

class SecureLogisticRegression:
    def __init__(self, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR):
        self.epochs = epochs
        self.lr = lr
        self.theta = None  # Model parameters
        self.secfx = mpc.SecFxp()
        
    def __approx_log__(self, x, terms=5):
        one = self.secfx(1)
        x_minus_1 = x - one
        result = self.secfx(0)
        sign = 1
        power = x_minus_1
        for n in range(1, terms + 1):
            term = power / n
            result += sign * term
            power *= x_minus_1
            sign *= -1
        return result

    def __approx_sigmoid__(self, x):
        # 5th-order Taylor approx: sigmoid(x) ≈ 0.5 + 0.25x - x³/48 + x⁵/480
        const_05 = self.secfx(0.5)
        const_025 = self.secfx(0.25)
        x3 = x * x * x
        x5 = x3 * x * x
        return const_05 + const_025 * x - (x3 / 48) + (x5 / 480)

    async def fit(self, X_parts, y_parts):
        """Securely train logistic regression using gradient descent.

        Args:
            X_parts (List[List[List[secfx]]]): List of X matrices from parties (all combined).
            y_parts (List[List[secfx]]): List of y vectors from parties (all combined).
        """
        
        # Concatenate data from all parties (already flattened)
        X = X_parts[0]  # shape: (n_samples, n_features)
        y = y_parts[0]  # shape: (n_samples,)
        n_samples = len(y)
        n_features = len(X[0])

        print(f"[Party {mpc.pid}] ✅ Loaded {n_samples} samples, {n_features} features")

        # Initialize theta (model weights) and bias to zeros
        theta = [self.secfx(0) for _ in range(n_features)]
        bias = self.secfx(0)
        lr_sec = self.secfx(self.lr)

        print(f"\n[Party {mpc.pid}] 🔎 Start logistic regression with {self.epochs} iterations and learning rate {self.lr}")
        for epoch in range(self.epochs):
            # Compute predictions: sigmoid(X @ theta)
            y_pred = [self.__approx_sigmoid__(sum(x_i[j] * theta[j] for j in range(n_features)) + bias) for x_i in X]
            
            # Compute error: y_pred - y
            error = [y_pred[i] - y[i] for i in range(n_samples)]

            # Compute gradients
            gradients = [
                sum(error[i] * X[i][j] for i in range(n_samples)) / n_samples
                for j in range(n_features)
            ]
            
            # Compute gradient for bias
            grad_bias = sum(error) / n_samples

            # Update theta and bias
            theta = [theta[j] - lr_sec * gradients[j] for j in range(n_features)]
            bias = bias - lr_sec * grad_bias

            # Debug: Print theta every 10 iterations
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                theta_debug = await mpc.output(theta + [bias])
                epsilon = self.secfx(1e-3)
                y_pred_clamped = [mpc.max(epsilon, mpc.min(1 - epsilon, yp)) for yp in y_pred]
                loss_terms = [
                    y[i] * self.__approx_log__(y_pred_clamped[i]) + (1 - y[i]) * self.__approx_log__(1 - y_pred_clamped[i])
                    for i in range(n_samples)
                ]
                loss = -sum(loss_terms) / n_samples
                loss_val = await mpc.output(loss)
                print(f"[Party {mpc.pid}] 🧮 Epoch {epoch + 1}: theta = {[float(t) for t in theta_debug]} | loss = {loss_val}")

        # Reveal final model weights
        print(f"\n[Party {mpc.pid}] ⌛ Reaching final training epoch...")
        try:
            theta_open = await mpc.output(theta + [bias])
            self.theta = [float(t) for t in theta_open]
            print(f"[Party {mpc.pid}] ✅ Training complete. Model weights: {self.theta}")
        except Exception as e:
            print(f"[Party {mpc.pid}] ❗ ERROR during mpc.output: {e}")
            self.theta = []
            self.bias = 0.0

    async def predict(self, X_input):
        """Securely predict using the trained model.

        Args:
            X_input (List[List[secfx]]): New input data (securely shared, same format).

        Returns:
            List[int]: Binary predictions (0 or 1).
        """
        if self.theta is None:
            raise ValueError("Model not trained. Call fit() before predict().")

        # Convert public model params back into secure fixed-point values
        secfx_theta = [self.secfx(w) for w in self.theta]

        sigmoid_outputs = []
        for x in X_input:
            dot = sum(a * b for a, b in zip(x, secfx_theta))
            sigmoid = self.__approx_sigmoid__(dot)
            sigmoid_outputs.append(await mpc.output(sigmoid))

        y_pred = [1 if p >= 0.5 else 0 for p in sigmoid_outputs]
        return y_pred

