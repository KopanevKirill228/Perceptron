import numpy as np


class SingleLayerPerceptron:
    def __init__(
        self,
        n_features,
        random_state=42,
        init_type="small_random",
        loss_type="bce",
        l2_lambda=0.0
    ):
        np.random.seed(random_state)

        if init_type == "zeros":
            self.w = np.zeros(n_features)

        elif init_type == "small_random":
            self.w = np.random.randn(n_features) * 0.01

        elif init_type == "large_random":
            self.w = np.random.normal(0, 10, n_features)

        else:
            raise ValueError("Unknown init_type. Use: zeros, small_random, large_random")

        self.b = 0.0

        self.loss_type = loss_type
        self.l2_lambda = l2_lambda

        self.train_losses = []
        self.val_losses = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward_linear(self, X):
        return np.dot(X, self.w) + self.b

    def forward(self, X):
        z = self.forward_linear(X)

        if self.loss_type == "bce":
            return self.sigmoid(z)

        elif self.loss_type == "hinge":
            return z

        else:
            raise ValueError("Unknown loss_type. Use: bce or hinge")

    def compute_bce_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

        l2_penalty = self.l2_lambda * np.sum(self.w ** 2) / 2

        return loss + l2_penalty

    def compute_hinge_loss(self, y_true, scores):
        margins = 1 - y_true * scores
        loss = np.mean(np.maximum(0, margins))

        l2_penalty = self.l2_lambda * np.sum(self.w ** 2) / 2

        return loss + l2_penalty

    def compute_loss(self, y_true, y_pred):
        if self.loss_type == "bce":
            return self.compute_bce_loss(y_true, y_pred)

        elif self.loss_type == "hinge":
            return self.compute_hinge_loss(y_true, y_pred)

        else:
            raise ValueError("Unknown loss_type. Use: bce or hinge")

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        lr=0.1,
        batch_size=32,
        use_momentum=False,
        beta=0.9
    ):
        n_samples = X_train.shape[0]

        v_w = np.zeros_like(self.w)
        v_b = 0.0

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size

                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                if self.loss_type == "bce":
                    y_pred = self.forward(X_batch)

                    error = y_pred - y_batch

                    dw = np.dot(X_batch.T, error) / len(y_batch)
                    db = np.mean(error)

                    dw += self.l2_lambda * self.w

                elif self.loss_type == "hinge":
                    scores = self.forward(X_batch)

                    margins = y_batch * scores
                    mask = margins < 1

                    if np.any(mask):
                        dw = -np.dot(X_batch[mask].T, y_batch[mask]) / len(y_batch)
                        db = -np.mean(y_batch[mask])
                    else:
                        dw = np.zeros_like(self.w)
                        db = 0.0

                    dw += self.l2_lambda * self.w

                else:
                    raise ValueError("Unknown loss_type. Use: bce or hinge")

                if use_momentum:
                    v_w = beta * v_w + dw
                    v_b = beta * v_b + db

                    self.w -= lr * v_w
                    self.b -= lr * v_b
                else:
                    self.w -= lr * dw
                    self.b -= lr * db

            train_pred = self.forward(X_train)
            val_pred = self.forward(X_val)

            train_loss = self.compute_loss(y_train, train_pred)
            val_loss = self.compute_loss(y_val, val_pred)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

    def predict(self, X):
        if self.loss_type == "bce":
            y_pred = self.forward(X)
            return (y_pred >= 0.5).astype(int)

        elif self.loss_type == "hinge":
            scores = self.forward(X)
            return np.where(scores >= 0, 1, -1)

        else:
            raise ValueError("Unknown loss_type. Use: bce or hinge")