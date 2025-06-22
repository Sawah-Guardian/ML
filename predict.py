from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import os
import cv2
import requests

app = Flask(__name__)
CORS(app)

model = YOLO('model/daun_padi.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'Preflight check successful'}), 200

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    os.makedirs('uploads', exist_ok=True)
    original_image_path = os.path.join('uploads', image_file.filename)
    image_file.save(original_image_path)

    try:
        results = model.predict(source=original_image_path, conf=0.6, save=False)
        pred_result = []
        predicted_image_path = None

        for r in results:
            im_array = r.plot()
            base_filename = os.path.basename(original_image_path)
            predicted_filename = "predicted_" + base_filename
            predicted_image_path = os.path.join('uploads', predicted_filename)

            cv2.imwrite(predicted_image_path, im_array)

            for box in r.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[cls_id]
                pred_result.append({
                    'class_id': cls_id,
                    'confidence': round(confidence, 2),
                    'class_name': class_name
                })

        if pred_result and predicted_image_path:
            detected_class = pred_result[0]['class_name']
            status_padi, info, rekomendasi = "", "", ""

            if detected_class == 'Bacteria_Leaf_Blight':
                status_padi = "Tidak Sehat"
                info = "Disebabkan oleh bakteri Xanthomonas oryzae..."
                rekomendasi = "Gunakan benih tahan, atur jarak tanam..."
            elif detected_class == 'Brown_Spot':
                status_padi = "Tidak Sehat"
                info = "Disebabkan oleh jamur Cochliobolus miyabeanus..."
                rekomendasi = "Gunakan fungisida dan sanitasi lahan..."
            elif detected_class == 'Leaf_smut':
                status_padi = "Tidak Sehat"
                info = "Disebabkan oleh jamur Entyloma oryzae..."
                rekomendasi = "Jaga kebersihan lahan, gunakan varietas tahan."
            else:
                status_padi = "Sehat"
                info = "Informasi detail untuk kelas ini belum tersedia."
                rekomendasi = "Konsultasikan dengan ahli pertanian."

            api_url = 'https://tkj-3b.com/tkj-3b.com/opengate/scan.php'
            user_id = '1'
            payload = {
                'user_id': user_id,
                'detected_class': detected_class,
                'status_padi': status_padi,
                'info': info,
                'rekomendasi': rekomendasi
            }

            try:
                with open(predicted_image_path, 'rb') as image_file_to_send:
                    files = {'image': (os.path.basename(predicted_image_path), image_file_to_send)}
                    response = requests.post(api_url, data=payload, files=files)
                    print("Response dari API eksternal:", response.status_code)
            except requests.exceptions.RequestException as e:
                print("Error saat mengirim ke API eksternal:", e)

        return jsonify({
            'message': 'Prediction successful',
            'predictions': pred_result,
            'predicted_image_url': f"/uploads/{os.path.basename(predicted_image_path)}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
