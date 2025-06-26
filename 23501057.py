import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QLineEdit, QPushButton, QComboBox, QListWidget, QFileDialog,
    QTextEdit, QTabWidget
)
from deep_learning import ModelBuilder, Trainer, dense_layer, conv2d_layer, lstm_layer, gru_layer, load_mnist

class NeuralNetGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Custom Neural Network GUI")

        self.layer_list = []
        self.model_builder = None
        self.trainer = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # === Architecture Section (Top) ===
        arch_layout = QVBoxLayout()
        layer_label = QLabel("🔧 Add Network Layers:")
        self.layer_type_combo = QComboBox()
        self.layer_type_combo.addItems(["Dense", "Conv2D", "LSTM", "GRU"])

        self.units_input = QLineEdit()
        self.units_input.setPlaceholderText("Units or Filters (e.g., 64)")

        self.kernel_input = QLineEdit()
        self.kernel_input.setPlaceholderText("Kernel Size (e.g., 3x3)")

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["relu", "sigmoid", "tanh"])

        self.dropout_input = QLineEdit()
        self.dropout_input.setPlaceholderText("Dropout Rate (e.g., 0.3, optional)")

        self.add_layer_btn = QPushButton("➕ Add Layer")
        self.add_layer_btn.clicked.connect(self.add_layer)

        layer_inputs = QHBoxLayout()
        for widget in [self.layer_type_combo, self.units_input, self.kernel_input,
                       self.activation_combo, self.dropout_input, self.add_layer_btn]:
            layer_inputs.addWidget(widget)

        self.layer_display = QListWidget()
        self.build_model_btn = QPushButton("🚧 Build Model")
        self.build_model_btn.clicked.connect(self.build_model)

        arch_layout.addWidget(layer_label)
        arch_layout.addLayout(layer_inputs)
        arch_layout.addWidget(QLabel("📋 Layer List:"))
        arch_layout.addWidget(self.layer_display)
        arch_layout.addWidget(self.build_model_btn)

        # === Training Section (Bottom) ===
        train_layout = QVBoxLayout()

        self.train_btn = QPushButton("🚀 Train")
        self.train_btn.clicked.connect(self.train_model)

        self.plot_btn = QPushButton("📈 Show Training Curves")
        self.plot_btn.clicked.connect(self.plot_history)

        self.save_btn = QPushButton("💾 Save Model")
        self.save_btn.clicked.connect(self.save_model)

        self.load_btn = QPushButton("📂 Load Model")
        self.load_btn.clicked.connect(self.load_model)

        ops_layout = QHBoxLayout()
        for btn in [self.train_btn, self.plot_btn, self.save_btn, self.load_btn]:
            ops_layout.addWidget(btn)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setPlaceholderText("Training results and logs will appear here...")

        train_layout.addLayout(ops_layout)
        train_layout.addWidget(QLabel("🧾 Results / Logs:"))
        train_layout.addWidget(self.result_display)

        # Combine architecture + training into main layout
        main_layout.addLayout(arch_layout)
        main_layout.addSpacing(20)
        main_layout.addLayout(train_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def add_layer(self):
        layer_type = self.layer_type_combo.currentText()
        units = int(self.units_input.text()) if self.units_input.text().isdigit() else 32
        activation = self.activation_combo.currentText()
        dropout = float(self.dropout_input.text()) if self.dropout_input.text() else None

        if layer_type == "Dense":
            block = dense_layer(units, activation, dropout)
        elif layer_type == "Conv2D":
            kernel = tuple(map(int, self.kernel_input.text().split("x"))) if "x" in self.kernel_input.text() else (3, 3)
            block = conv2d_layer(units, kernel, activation)
        elif layer_type == "LSTM":
            block = lstm_layer(units)
        elif layer_type == "GRU":
            block = gru_layer(units)
        else:
            return

        self.layer_list.append((layer_type, block))
        self.layer_display.addItem(f"{layer_type} | {units} | {activation}{' | Dropout: ' + str(dropout) if dropout else ''}")

    def build_model(self):
        input_shape = (28, 28, 1)
        num_classes = 10
        self.model_builder = ModelBuilder(input_shape, num_classes)

        for _, block in self.layer_list:
            self.model_builder.add_block(block)

        self.model_builder.finalize()
        self.model_builder.compile()
        self.result_display.append("✅ Model has been successfully compiled.")

    def train_model(self):
        x_train, y_train, x_val, y_val, _, _ = load_mnist()
        model = self.model_builder.get_model()
        self.trainer = Trainer(model)
        history = self.trainer.train(x_train, y_train, x_val, y_val, epochs=5)

        self.result_display.append("🚀 Training Complete!")
        acc = history.history.get("val_accuracy", [0])[-1]
        loss = history.history.get("val_loss", [0])[-1]
        self.result_display.append(f"📊 Final Validation Accuracy: {acc:.4f}")
        self.result_display.append(f"📉 Final Validation Loss: {loss:.4f}")

    def plot_history(self):
        if self.trainer:
            self.trainer.plot_training_curves()
            self.result_display.append("📈 Training curves plotted.")

    def save_model(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Your Model", "", "HDF5 Files (*.h5)")
        if path:
            self.model_builder.get_model().save(path)
            self.result_display.append(f"💾 Model saved at: {path}")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load a Saved Model", "", "HDF5 Files (*.h5)")
        if path:
            from tensorflow.keras.models import load_model
            model = load_model(path)
            self.model_builder = ModelBuilder((28, 28, 1), 10)
            self.trainer = Trainer(model)
            self.result_display.append("📂 Model loaded successfully.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = NeuralNetGUI()
    gui.show()
    sys.exit(app.exec())
