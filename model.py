# app.py

from flask import Flask, request, render_template, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

# Adım 2'de oluşturduğumuz model.py dosyasından HybridQNN sınıfını içe aktar
from model import HybridQNN, N_QUBITS, N_LAYERS

# --- Uygulama ve Model Kurulumu ---
app = Flask(__name__)

# Modeli ve gerekli bilgileri yükle (sadece bir kez, başlangıçta)
N_CLASSES = 3
# Sunucu ortamlarında genellikle GPU bulunmaz, bu yüzden CPU'ya zorluyoruz.
# map_location=PROCESSOR, modelin CPU üzerinde yüklenmesini sağlar.
PROCESSOR = torch.device("cpu") 

# Modelin iskeletini oluştur
model = HybridQNN(n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=N_CLASSES)
# Eğitilmiş ağırlıkları yükle
model.load_state_dict(torch.load('en_iyi_model.pth', map_location=PROCESSOR))
model.eval() # Modeli değerlendirme moduna al

# Gelen resimlere uygulanacak dönüşümler
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Sınıf isimleri
class_names = ['paper', 'rock', 'scissors']

# --- Web Sayfası Rotaları (Endpoints) ---

@app.route('/', methods=['GET'])
def home():
    """
    Ana sayfayı (index.html) kullanıcıya gösterir.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Kullanıcıdan gelen resim dosyasını alır, model ile tahmin yapar
    ve sonucu JSON formatında geri döndürür.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
    
    file = request.files['file']
    
    try:
        # Gelen dosyayı bir resim olarak aç
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Resmi modele uygun formata getir
        image_tensor = inference_transforms(image).unsqueeze(0)
        
        # Tahmin yap
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probabilities, 1)
            
            pred_class = class_names[top_idx.item()]
            confidence = top_prob.item()
            
        # Sonucu JSON formatında geri döndür
        return jsonify({'prediction': pred_class, 'confidence': f"{confidence*100:.2f}"})

    except Exception as e:
        # Bir hata oluşursa, hatayı JSON olarak bildir
        return jsonify({'error': str(e)}), 500

# NOT: Dağıtım (deployment) ortamlarında, sunucuyu Gunicorn gibi bir WSGI sunucusu
# başlatır. Bu yüzden, geliştirme için kullanılan aşağıdaki blok kaldırılmıştır.
# if __name__ == '__main__':
#     app.run(debug=True)
