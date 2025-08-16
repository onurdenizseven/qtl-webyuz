# model.py

import torch
import torch.nn as nn
from torchvision import models
import pennylane as qml

# --- Model Mimarisi için Gerekli Sabitler ve Tanımlar ---

N_QUBITS = 4
N_LAYERS = 6
DEVICE = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(DEVICE, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.templates.BasicEntanglerLayers(weights, wires=range(N_QUBITS), rotation=qml.RY)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Hibrit Kuantum-Klasik Model Sınıfı
class HybridQNN(nn.Module):
    def __init__(self, n_qubits, n_layers, n_classes):
        super(HybridQNN, self).__init__()
        
        # --- ÖNEMLİ DEĞİŞİKLİK BURADA ---
        # İnternetten önceden eğitilmiş ağırlıkları indirmesini engelliyoruz.
        # Sadece modelin mimarisini oluşturuyoruz. Ağırlıkları zaten 
        # 'en_iyi_model.pth' dosyasından kendimiz yükleyeceğiz.
        self.feature_extractor = models.resnet18(weights=None)
        
        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()

        self.pre_quantum = nn.Linear(num_features, n_qubits)

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        self.classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pre_quantum(x)
        x = self.quantum_layer(x)
        x = self.classifier(x)
        return x
