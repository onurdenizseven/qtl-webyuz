# model.py

import torch
import torch.nn as nn
from torchvision import models
import pennylane as qml

# --- Model Mimarisi için Gerekli Sabitler ve Tanımlar ---

# Kuantum Devresi için temel hiperparametreler
N_QUBITS = 4
N_LAYERS = 6
DEVICE = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(DEVICE, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    Bu fonksiyon, klasik veriyi kuantum durumlarına kodlayan ve
    eğitilebilir kuantum katmanlarını uygulayan kuantum devresini tanımlar.
    """
    # Girdileri (klasik özellikler) qubit'lerin açılarına kodla
    qml.templates.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    
    # Eğitilebilir, dolaşıklık yaratan kuantum katmanları
    qml.templates.BasicEntanglerLayers(weights, wires=range(N_QUBITS), rotation=qml.RY)
    
    # Her bir qubit'ten ölçüm alarak sonucu klasik dünyaya geri döndür
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Hibrit Kuantum-Klasik Model Sınıfı
class HybridQNN(nn.Module):
    """
    Bu sınıf, önceden eğitilmiş bir ResNet18 modelini (özellik çıkarıcı olarak)
    ve bir kuantum devresini (sınıflandırıcı olarak) birleştirir.
    """
    def __init__(self, n_qubits, n_layers, n_classes):
        super(HybridQNN, self).__init__()
        
        # 1. Klasik Özellik Çıkarıcı (Pre-trained ResNet18)
        self.feature_extractor = models.resnet18(pretrained=True)
        num_features = self.feature_extractor.fc.in_features
        # ResNet'in son sınıflandırma katmanını kaldırıyoruz
        self.feature_extractor.fc = nn.Identity()

        # 2. Kuantum Devresine Adaptör Katmanı
        # ResNet çıktısını (512 özellik) qubit sayısına (4) düşüren lineer katman
        self.pre_quantum = nn.Linear(num_features, n_qubits)

        # 3. Kuantum Katmanı
        # PennyLane'in Torch ile uyumlu kuantum katmanı
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # 4. Son Klasik Sınıflandırıcı
        # Kuantum devresi çıktısını (4 değer) nihai sınıf sayısına (3) eşleyen katman
        self.classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        """
        Bir verinin modelden geçerken izlediği yol.
        """
        x = self.feature_extractor(x)
        x = self.pre_quantum(x)
        x = self.quantum_layer(x)
        x = self.classifier(x)
        return x
