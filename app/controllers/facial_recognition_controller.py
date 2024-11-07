from flask import Blueprint, request, jsonify
from app.services.facial_recognition_service import FacialRecognitionService

facial_recognition_bp = Blueprint('facial_recognition', __name__)

@facial_recognition_bp.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    face_data = data.get('face_data')
    if not face_data:
        return jsonify({"error": "No face data provided"}), 400

    # Llama al servicio para procesar los datos faciales
    response = FacialRecognitionService.process_face_data(face_data)
    return jsonify(response)
