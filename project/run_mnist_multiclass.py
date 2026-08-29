from pathlib import Path

from mnist import MNIST

import minitorch

MNIST_DATA_DIR = Path(__file__).resolve().parent / "data"
_mnist_training_data = None

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
BACKEND = FastTensorBackend
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def RParam(*shape, backend=BACKEND):
    r = 0.1 * (minitorch.rand(shape, backend=backend) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size, backend=BACKEND):
        super().__init__()
        self.weights = RParam(in_size, out_size, backend=backend)
        self.bias = RParam(out_size, backend=backend)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw, backend=BACKEND):
        super().__init__()
        self.weights = RParam(
            out_channels, in_channels, kh, kw, backend=backend
        )
        self.bias = RParam(out_channels, 1, 1, backend=backend)

    def forward(self, input):
        # ASSIGN4.5
        out = minitorch.conv2d(input, self.weights.value) + self.bias.value
        return out
        # END ASSIGN4.5


class Network(minitorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.

    This model should implement the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """

    def __init__(self, backend=BACKEND):
        super().__init__()
        self.backend = backend

        # For vis
        self.mid = None
        self.out = None

        # ASSIGN4.5
        self.conv1 = Conv2d(1, 4, 3, 3, backend)
        self.conv2 = Conv2d(4, 8, 3, 3, backend)
        self.linear1 = Linear(392, 64, backend)
        self.linear2 = Linear(64, C, backend)
        # END ASSIGN4.5

    def forward(self, x):
        # ASSIGN4.5
        batch_size = x.shape[0]
        x = self.conv1(x).relu()
        self.mid = x
        x = self.conv2(x).relu()
        self.out = x
        x = minitorch.avgpool2d(x, (4, 4))
        x = self.linear1(x.view(batch_size, 392)).relu()
        x = minitorch.dropout(x, 0.25, ignore=not self.training)
        x = self.linear2(x)
        x = minitorch.logsoftmax(x, dim=1)
        return x
        # END ASSIGN4.5


def make_mnist(start, stop):
    global _mnist_training_data
    if _mnist_training_data is None:
        try:
            _mnist_training_data = MNIST(str(MNIST_DATA_DIR)).load_training()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"MNIST files were not found in {MNIST_DATA_DIR}. "
                "Run `mnist_get_data.sh` from that directory first."
            ) from exc

    images, labels = _mnist_training_data
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


def default_log_fn(epoch, total_loss, correct, total, losses, model):
    print(f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}")


class ImageTrain:
    def __init__(self, backend=BACKEND):
        self.backend = backend
        self.model = Network(backend)

    def run_one(self, x):
        return self.model.forward(
            minitorch.tensor([x], backend=self.backend).view(1, 1, H, W)
        )

    def train(
        self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network(self.backend)
        model = self.model
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, BATCH)
            ):
                batch_size = min(BATCH, n_training_samples - example_num)
                optim.zero_grad()
                y = minitorch.tensor(
                    y_train[example_num : example_num + batch_size],
                    backend=self.backend,
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + batch_size],
                    backend=self.backend,
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x.view(batch_size, 1, H, W)).view(batch_size, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.backend == self.backend
                loss.view(1).backward()

                total_loss += loss[0]
                losses.append(total_loss)

                # Update
                optim.step()

                if batch_num % 5 == 0:
                    model.eval()
                    # Evaluate on one held-out batch.
                    correct = 0
                    validation_size = min(BATCH, len(X_val))
                    if validation_size > 0:
                        y = minitorch.tensor(
                            y_val[:validation_size],
                            backend=self.backend,
                        )
                        x = minitorch.tensor(
                            X_val[:validation_size],
                            backend=self.backend,
                        )
                        out = model.forward(
                            x.view(validation_size, 1, H, W)
                        ).view(validation_size, C)
                        for i in range(validation_size):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    log_fn(
                        epoch,
                        total_loss,
                        correct,
                        validation_size,
                        losses,
                        model,
                    )

                    total_loss = 0.0
                    model.train()


if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))
    ImageTrain().train(data_train, data_val, learning_rate=0.01)
