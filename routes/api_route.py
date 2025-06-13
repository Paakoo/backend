from flask import Blueprint, request, jsonify, render_template
from flask_jwt_extended import jwt_required
from werkzeug.utils import secure_filename
import os
import time
from datetime import datetime
from services.database import save_user, get_user_by_credentials, save_presence, check_presence, get_history, get_employees, delete_employee, get_employee_by_id,get_db_connection
from services.face_recognition import load_h5_embeddings, find_matching_face, update_dataset
from utils.image_processing import crop_and_save_face
from utils.file import allowed_file
from services.auth import get_current_user
from config.settings import Config
from deepface import DeepFace
from retinaface import RetinaFace
import cv2
import numpy as np
from mtcnn import MTCNN

api_route_bp = Blueprint('api_route_bp', __name__)

@api_route_bp.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({'status': 'API is running'})

@api_route_bp.route('/capture')
def capture():
    return render_template('FaceCapture.html')

@api_route_bp.route('/api/save_image', methods=['POST'])
def save_image_route():
    data = request.get_json()
    angle = data.get('angle')
    count = data.get('count')
    image_data = data.get('image')
    username = data.get('username')

    user_folder = os.path.join(Config.BASE_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)

    try:
        filename = f"{username}_{angle}_{count}.jpg"
        processed_filename = crop_and_save_face(image_data, user_folder, filename)
        
        if processed_filename:
            return jsonify({
                'status': 'success',
                'message': f'Image saved as {processed_filename}',
                'filename': processed_filename
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to process image'
            }), 500
    except Exception as e:
        print(f"Error saving image: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Error saving image'
        }), 500

@api_route_bp.route('/api/save_user', methods=['POST'])
def save_user_route():
    data = request.get_json()
    username = data.get('name')
    email = data.get('email')
    role = data.get('role')

    if not username or not email or not role:
        return jsonify({'error': 'All fields are required'}), 400

    response, status = save_user(username, email, role)
    return jsonify(response), status

location_data = {}

@api_route_bp.route('/api/getlocation', methods=['POST'])
def get_location():
    try:
        data = request.get_json()
        global location_data
        location_data = {
            'location_type': data.get('location_data', {}).get('location_type'),
            'office_name': data.get('location_data', {}).get('office_name'),
            'timestamp': data.get('location_data', {}).get('timestamp'),
            'latitude': data.get('location_data', {}).get('latitude'),
            'longitude': data.get('location_data', {}).get('longitude')
        }
        return jsonify({
            'status': 'success',
            'message': 'Location data received',
            'data': location_data
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@api_route_bp.route('/api/present', methods=['POST'])
@jwt_required()
def present():
    start_time = time.time()
    temp_path = None

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({
                'status': 'error',
                'message': 'Could not identify user'
            }), 401

        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image file provided'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
        
        # Get form data
        location_type = request.form.get('location_type')
        office_name = request.form.get('office_name')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        timestamp = request.form.get('timestamp')

        if not all([location_type, office_name, latitude, longitude, timestamp]):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields'
            }), 400
        
        filename = secure_filename(file.filename)
        temp_path = os.path.join(Config.UPLOAD_FOLDER, f"temp_{filename}")
        file.save(temp_path)

        spoof_result = DeepFace.extract_faces(img_path=temp_path, anti_spoofing=True)
        if not spoof_result or len(spoof_result) == 0:
            return jsonify({
                'status': 'error',
                'message': 'No face detected',
                'data': {'error_code': 'no_face_detected'}
            }), 400

        face_data = spoof_result[0]
        is_real = face_data.get('is_real', False)
        spoof_confidence = float(face_data.get('confidence', 0))

        if not is_real:
            return jsonify({
                'status': 'error',
                'message': 'Spoof detected. Please use a real face.',
                'data': {'error_code': 'spoof_detected'}
            }), 400

        image = cv2.imread(temp_path)
        faces = RetinaFace.detect_faces(image)
        if not faces:
            return jsonify({
                'status': 'error',
                'message': 'No face detected in image',
                'data': {'error_code': 'no_face_detected'}
            }), 400

        embedding_obj = DeepFace.represent(
            img_path=temp_path,
            model_name=Config.FACE_RECOGNITION_MODEL,
            detector_backend=Config.FACE_DETECTION_MODEL,
            enforce_detection=True,
            align=True
        )

        embeddings = load_h5_embeddings()
        current_embedding = np.array(embedding_obj[0]['embedding'])
        matched_name, similarity = find_matching_face(current_embedding, embeddings)

        if not matched_name:
            response_data = {
                'status': 'error',
                'message': 'No matching face found in database',
                'data': {
                    'error_code': 'face_not_found',
                    'matched_name': None,
                    'confidence': None
                }
            }
            print(response_data)
            return jsonify(response_data), 404

        if matched_name != current_user:
            response_data = {
                'status': 'error',      
                'message': f"Face matched with different user: {matched_name}",
                'data': {
                    'error_code': 'face_mismatch',
                    'matched_name': matched_name,
                    'confidence': float(similarity)
                }
            }
            print(response_data)
            return jsonify(response_data), 400

        response_data = {
            'status': 'success',
            'message': 'Face matched with authenticated user',
            'data': {
                'matched_name': matched_name,
                'confidence': float(similarity),
                'database_name': current_user,
                'is_real': is_real,
                'spoof_confidence': spoof_confidence
            },
            'timing': {
                'total': f"{time.time() - start_time:.3f}s"
            },
            'location': {
                'location_type': location_type,
                'office_name': office_name,
                'latitude': latitude,
                'longitude': longitude,
                'timestamp': timestamp,
            },
        }
        print(response_data)
        return jsonify(response_data), 200
    except Exception as e:
        response_data = {
            'status': 'error',
            'message': 'Face could not be detected in face recognition.',
            'data': {'error_code': 'no_face_detected'}
        }
        print(response_data)
        return jsonify(response_data),  400
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")

@api_route_bp.route('/api/update_dataset', methods=['POST'])
def update_dataset_route():
    response, status = update_dataset()
    return jsonify(response), status

@api_route_bp.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")
        
        if not email or not password:
            return jsonify({
                'status': 'error',
                'message': 'Email and password are required'
            }), 400
            
        user = get_user_by_credentials(email, password)
        
        if user:
            access_token = create_access_token(
                identity=user['nama_karyawan'],
                expires_delta=False
            )
            response = {
                'status': 'success',
                'message': 'Login successful',
                'data': {
                    'id_karyawan': user['id_karyawan'],
                    'email': user['email'],
                    'nama': user['nama_karyawan'],
                    'password': user['password'],
                    'token': access_token,
                    'token_type': 'Bearer',
                    'role': user['role'],
                    'expires_in': None
                }
            }
            return jsonify(response), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid email or password'
            }), 401
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_route_bp.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user = get_current_user()
    return jsonify(logged_in_as=current_user), 200

@api_route_bp.route('/api/presence', methods=['POST'])
def save_presence_route():
    json_data = request.get_json()
    location_data = json_data.get('location', {})
    data = json_data.get('data', {})
    
    presence_data = {
        'timestamp': location_data.get('timestamp'),
        'latitude': location_data.get('latitude'),
        'longitude': location_data.get('longitude'),
        'office_name': location_data.get('office_name'),
        'location_type': location_data.get('location_type'),
        'database_name': data.get('database_name')
    }
    
    response, status = save_presence(presence_data)
    return jsonify(response), status

@api_route_bp.route('/api/checkclockIn', methods=['GET'])
def check_presence_in():
    user_name = request.args.get('nama_karyawan')
    user_id = request.args.get('id_karyawan')
    
    if not user_name and not user_id:
        return jsonify({
            'status': 'error',
            'message': 'Parameter "name" atau "id" diperlukan untuk memeriksa presensi.'
        }), 400
    
    response, status = check_presence(user_name, user_id, 'in')
    return jsonify(response), status

@api_route_bp.route('/api/checkclockOut', methods=['GET'])
def check_presence_out():
    user_name = request.args.get('nama_karyawan')
    user_id = request.args.get('id_karyawan')
    
    if not user_name and not user_id:
        return jsonify({
            'status': 'error',
            'message': 'Parameter "name" atau "id" diperlukan untuk memeriksa presensi.'
        }), 400
    
    response, status = check_presence(user_name, user_id, 'out')
    return jsonify(response), status

@api_route_bp.route('/api/history', methods=['GET'])
def history():
    user_name = request.args.get('nama_karyawan')
    
    if not user_name:
        return jsonify({
            'status': 'error',
            'message': 'Parameter "nama_karyawan" diperlukan untuk melihat history.'
        }), 400
    
    response, status = get_history(user_name)
    return jsonify(response), status

@api_route_bp.route('/api/employees', methods=['GET'])
def get_employees_route():
    employees, status = get_employees()
    return jsonify(employees), status

@api_route_bp.route('/api/employees/<int:employee_id>', methods=['GET'])
def get_employee_by_id(employee_id):
    employee, status = get_employee_by_id(employee_id)
    return jsonify(employee), status

@api_route_bp.route('/api/employees/<int:id_karyawan>', methods=['DELETE'])
def delete_employee_route(id_karyawan):
    response, status = delete_employee(id_karyawan)
    if status == 200:
        h5_delete_result = delete_user_from_h5(Config.EMBEDDINGS_PATH, response['nama_karyawan'])
        response['h5_status'] = h5_delete_result
    return jsonify(response), status

@api_route_bp.route('/api/employees/<int:employee_id>', methods=['PUT'])
def update_employee(employee_id):
    data = request.json
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute("""
                UPDATE karyawan
                SET nama_karyawan = %s, email = %s, role = %s
                WHERE id_karyawan = %s
            """, (data.get("name"), data.get("email"), data.get("role"), employee_id))
            conn.commit()
            return jsonify({"id": employee_id, **data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@api_route_bp.route('/twins', methods=['POST'])
def twins():
    start_time = time.time()
    temp_path = None
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No image file provided'
            }), 400
            
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Empty file'
            }), 400
            
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type'
            }), 400
            
        filename = secure_filename(file.filename)
        temp_path = os.path.join(Config.UPLOAD_FOLDER, f"temp_{filename}")
        file.save(temp_path)
        
        spoof_start = time.time()
        spoof_result = DeepFace.extract_faces(
            img_path=temp_path,
            anti_spoofing=True
        )
        spoof_time = time.time() - spoof_start

        if not spoof_result or len(spoof_result) == 0:
            return jsonify({'status': 'error', 'message': 'No face detected'}), 400

        face_data = spoof_result[0]
        is_real = face_data.get('is_real', False)
        spoof_confidence = float(face_data.get('confidence', 0))

        detection_start = time.time()
        image = cv2.imread(temp_path)
        # detector = MTCNN()
        faces = RetinaFace.detect_faces(image)
        detection_time = time.time() - detection_start

        if faces:
            process_start = time.time()
            x, y, width, height = faces[0]['box']
            margin = 20
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            width += margin * 2
            height += margin * 2
            
            cropped_face = image[y:y+height, x:x+width]
            resized_face = cv2.resize(cropped_face, (250, 250))
            file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            cv2.imwrite(file_path, resized_face)
            process_time = time.time() - process_start

            embedding_start = time.time()
            embedding_obj = DeepFace.represent(
                img_path=file_path,
                model_name=Config.FACE_RECOGNITION_MODEL,
                detector_backend=Config.FACE_DETECTION_MODEL,
                enforce_detection=False,
                align=True
            )
            embedding_time = time.time() - embedding_start

            matching_start = time.time()
            embeddings = load_h5_embeddings()
            current_embedding = np.array(embedding_obj[0]['embedding'])
            
            matched_name, similarity = find_matching_face(current_embedding, embeddings)
            matching_time = time.time() - matching_start
            
            if matched_name:
                return jsonify({
                    'status': 'success',
                    'message': 'Face matched with authenticated user',
                    'data': {
                        'matched_name': matched_name,
                        'confidence': float(similarity),
                        'filename': filename,
                        'is_real': is_real,
                        'spoof_confidence': spoof_confidence
                    },
                    'timing': {
                        'spoofing': f"{spoof_time:.3f}s",
                        'detection': f"{detection_time:.3f}s",
                        'processing': f"{process_time:.3f}s",
                        'embedding': f"{embedding_time:.3f}s",
                        'matching': f"{matching_time:.3f}s",
                        'total': f"{time.time() - start_time:.3f}s"
                    }
                }), 200
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Face matched with different user',
                    'data': {
                        'matched_name': matched_name,
                        'confidence': float(similarity),
                        'filename': filename,
                        'is_real': is_real,
                        'spoof_confidence': spoof_confidence
                    },
                    'timing': {
                        'spoofing': f"{spoof_time:.3f}s",
                        'detection': f"{detection_time:.3f}s",
                        'processing': f"{process_time:.3f}s",
                        'embedding': f"{embedding_time:.3f}s",
                        'matching': f"{matching_time:.3f}s",
                        'total': f"{time.time() - start_time:.3f}s"
                    }
                }), 200

        return jsonify({
            'status': 'error',
            'message': 'No face match found'
        }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")

@api_route_bp.route('/api/attendance', methods=['GET'])
def get_attendance_data():
    try:
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')
        
        if not start_date or not end_date:
            return jsonify({
                'success': False,
                'data': [],
                'message': 'Parameter startDate dan endDate diperlukan'
            }), 400
        
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        conn = get_db_connection()
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            query = """
                SELECT 
                    *
                FROM 
                    absensi
                WHERE 
                    DATE(absensi_masuk) BETWEEN %s AND %s
                ORDER BY
                    absensi_masuk DESC, nama_karyawan ASC
            """
            cursor.execute(query, (start_date, end_date))
            records = cursor.fetchall()
            
            result = []
            for row in records:
                absensi_masuk = row['absensi_masuk'].isoformat() if row['absensi_masuk'] else None
                date = row['absensi_masuk'].date().isoformat() if row['absensi_masuk'] else None
                check_in_time = row['absensi_masuk'].strftime('%H:%M:%S') if row['absensi_masuk'] else None
                
                result.append({
                    'id': row['id_absensi'],
                    'date': date,
                    'employeeId': row['id_karyawan'],
                    'employeeName': row['nama_karyawan'],
                    'workType': row['work_type'],
                    'office': row['office'],
                    'location': {
                        'latitude': float(row['latitude']) if row['latitude'] is not None else None,
                        'longitude': float(row['longitude']) if row['longitude'] is not None else None
                    },
                    'checkInTime': check_in_time,
                    'status': 'Hadir'
                })
        
            return jsonify({
                'success': True,
                'data': result,
                'message': 'Data absensi berhasil diambil'
            }), 200
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'data': [],
            'message': f'Gagal mengambil data absensi: {str(e)}'
        }), 500
    finally:
        conn.close()

@api_route_bp.route('/api/dataemployee', methods=['GET'])
def dataemployee():
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            query = "SELECT id_karyawan, nama_karyawan, email, password, role FROM karyawan"
            cursor.execute(query)
            records = cursor.fetchall()
            
            result = [{
                'id': row['id_karyawan'],
                'name': row['nama_karyawan'],
            } for row in records]
            
            return jsonify({
                'success': True,
                'data': result,
                'message': 'Data karyawan berhasil diambil'
            }), 200
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'data': [],
            'message': f'Gagal mengambil data karyawan: {str(e)}'
        }), 500
    finally:
        conn.close()

@api_route_bp.route('/api/getattendance', methods=['GET'])
def get_attendance():
    today_date_str = request.args.get('todayDate')
    end_date_str = request.args.get('endDate')

    try:
        if not today_date_str or not end_date_str:
            return jsonify({"success": False, "message": "Both 'todayDate' and 'endDate' are required."}), 400

        today_date = datetime.strptime(today_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        conn = get_db_connection()
        with conn.cursor() as cursor:
            query = """
                SELECT id_karyawan, nama_karyawan, work_type, office,
                       latitude, longitude, absensi_masuk
                FROM absensi
                WHERE DATE(absensi_masuk) BETWEEN %s AND %s
                ORDER BY absensi_masuk ASC
            """
            cursor.execute(query, (today_date, end_date))
            rows = cursor.fetchall()

            attendance_data = []
            for row in rows:
                attendance_data.append({
                    "date": row[6].strftime('%Y-%m-%d'),
                    "checkInTime": row[6].strftime('%H:%M:%S'),
                    "employee": {
                        "employeeId": str(row[0]),
                        "name": row[1],
                        "office": row[3],
                    },
                    "status": row[2]
                })

            return jsonify({
                "success": True,
                "message": "Data absensi berhasil diambil",
                "data": attendance_data
            }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error occurred: {str(e)}"
        }), 500
    finally:
        conn.close()

@api_route_bp.route('/api/getattendanceMonth', methods=['GET'])
def get_attendance_month():
    start_date_str = request.args.get('startDate')
    end_date_str = request.args.get('endDate')

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        conn = get_db_connection()
        with conn.cursor() as cursor:
            query = """
                SELECT id_karyawan, nama_karyawan, work_type, office,
                       latitude, longitude, absensi_masuk
                FROM absensi
                WHERE DATE(absensi_masuk) BETWEEN %s AND %s
                ORDER BY absensi_masuk ASC
            """
            cursor.execute(query, (start_date, end_date))
            rows = cursor.fetchall()

            attendance_data = []
            for row in rows:
                attendance_data.append({
                    "date": row[6].strftime('%Y-%m-%d'),
                    "checkInTime": row[6].strftime('%H:%M:%S'),
                    "employee": {
                        "employeeId": str(row[0]),
                        "name": row[1],
                        "office": row[3],
                    },
                    "location": {
                        "lat": row[4],
                        "lng": row[5]
                    },
                    "workType": row[2]
                })

            return jsonify({"success": True, "data": attendance_data}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        conn.close()
        
        
@api_route_bp.route('/twinsdua', methods=['POST'])
def twinsdua():
    start_time = time.time()
    temp_path = Config.UPLOAD_FOLDER

    try:
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No image file provided'
            }), 400
            
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Empty file'
            }), 400
            
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type'
            }), 400

        # Save and process file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(Config.UPLOAD_FOLDER, f"temp_{filename}")
        file.save(temp_path)

        # Process image with detailed timing
        try:
            # Anti-spoofing check first with timing
            spoof_start = time.time()
            spoof_result = DeepFace.extract_faces(
                img_path=temp_path,
                anti_spoofing=True
            )
            spoof_time = time.time() - spoof_start

            if not spoof_result or len(spoof_result) == 0:
                return jsonify({
                    'status': 'error',
                    'message': 'No face detected',
                    'data': {'error_code': 'no_face_detected'}
                }), 400

            face_data = spoof_result[0]
            is_real = face_data.get('is_real', False)
            spoof_confidence = float(face_data.get('confidence', 0))

            if not is_real:
                return jsonify({
                    'status': 'error',
                    'message': 'Spoof detected. Please use a real face.',
                    'data': {
                        'error_code': 'spoof_detected',
                        'is_real': is_real,
                        'spoof_confidence': spoof_confidence
                    },
                    'timing': {
                        'spoofing': f"{spoof_time:.3f}s",
                        'total': f"{time.time() - start_time:.3f}s"
                    }
                }), 400

            # Continue with face detection if real face
            detection_start = time.time()
            image = cv2.imread(temp_path)
            detector = MTCNN()
            faces = detector.detect_faces(image)
            detection_time = time.time() - detection_start

            if not faces:
                return jsonify({
                    'status': 'error',
                    'message': 'No face detected in image',
                    'data': {'error_code': 'no_face_detected'},
                    'timing': {
                        'spoofing': f"{spoof_time:.3f}s",
                        'detection': f"{detection_time:.3f}s",
                        'total': f"{time.time() - start_time:.3f}s"
                    }
                }), 400

            # Time face processing
            process_start = time.time()
            x, y, width, height = faces[0]['box']
            margin = 20
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            width += margin * 2
            height += margin * 2
            
            cropped_face = image[y:y+height, x:x+width]
            resized_face = cv2.resize(cropped_face, (250, 250))
            file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            cv2.imwrite(file_path, resized_face)
            process_time = time.time() - process_start
            
            # Continue with embedding generation and matching with timing
            embedding_start = time.time()
            embedding_obj = DeepFace.represent(
                img_path=file_path,
                model_name=Config.FACE_RECOGNITION_MODEL,
                detector_backend=Config.FACE_DETECTION_MODEL,
                enforce_detection=False,  # Changed to False since we already detected face
                align=True
            )
            embedding_time = time.time() - embedding_start

            # Time matching
            matching_start = time.time()
            embeddings = load_h5_embeddings()
            current_embedding = np.array(embedding_obj[0]['embedding'])
            matched_name, similarity = find_matching_face(current_embedding, embeddings)
            matching_time = time.time() - matching_start
            os.remove(file_path)
            if not matched_name:
                response_data = {
                    'status': 'error',
                    'message': 'No matching face found in database',
                    'data': {
                        'error_code': 'face_not_found',
                        'matched_name': None,
                        'confidence': None,
                        'is_real': is_real,
                        'spoof_confidence': spoof_confidence
                    },
                    'timing': {
                        'spoofing': f"{spoof_time:.3f}s",
                        'detection': f"{detection_time:.3f}s",
                        'processing': f"{process_time:.3f}s",
                        'embedding': f"{embedding_time:.3f}s",
                        'matching': f"{matching_time:.3f}s",
                        'total': f"{time.time() - start_time:.3f}s"
                    }
                }
                print(response_data)
                return jsonify(response_data), 404

            # Success response with detailed timing
            response_data = {
                'status': 'success',
                'message': 'Face recognition successful',
                'data': {
                    'matched_name': matched_name,
                    'confidence': float(similarity),
                    'filename': filename,
                    'is_real': is_real,
                    'spoof_confidence': spoof_confidence
                },
                'timing': {
                    'spoofing': f"{spoof_time:.3f}s",
                    'detection': f"{detection_time:.3f}s",
                    'processing': f"{process_time:.3f}s",
                    'embedding': f"{embedding_time:.3f}s",
                    'matching': f"{matching_time:.3f}s",
                    'total': f"{time.time() - start_time:.3f}s"
                }
            }
            print(response_data)
            return jsonify(response_data), 200

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    except Exception as e:
        response_data = {
           'status': 'error',
            'message': 'Face could not be detected in face recognition.',
            'data': {'error_code': 'no_face_detected'}
        }
        print(response_data)
        return jsonify(response_data), 400

    finally:
        # Clean up temp files
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")
                
@api_route_bp.route('/twinstiga', methods=['POST'])
def twinstiga():
    start_time = time.time()
    temp_path = None

    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image file provided'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
        
        # Get form data
        # location_type = request.form.get('location_type')
        # office_name = request.form.get('office_name')
        # latitude = request.form.get('latitude')
        # longitude = request.form.get('longitude')
        # timestamp = request.form.get('timestamp')

        # if not all([location_type, office_name, latitude, longitude, timestamp]):
        #     return jsonify({
        #         'status': 'error',
        #         'message': 'Missing required fields'
        #     }), 400
        
        filename = secure_filename(file.filename)
        temp_path = os.path.join(Config.UPLOAD_FOLDER, f"temp_{filename}")
        file.save(temp_path)

        spoof_result = DeepFace.extract_faces(img_path=temp_path, anti_spoofing=True)
        # if not spoof_result or len(spoof_result) == 0:
        #     return jsonify({
        #         'status': 'error',
        #         'message': 'No face detected',
        #         'data': {'error_code': 'no_face_detected'}
        #     }), 400

        face_data = spoof_result[0]
        is_real = face_data.get('is_real', False)
        spoof_confidence = float(face_data.get('confidence', 0))

        if not is_real:
            return jsonify({
                'status': 'error',
                'message': 'Spoof detected. Please use a real face.',
                'data': {'error_code': 'spoof_detected'}
            }), 400

        image = cv2.imread(temp_path)
        detector = RetinaFace
        faces = detector.detect_faces(image)
        if not faces:
            return jsonify({
                'status': 'error',
                'message': 'No face detected in image',
                'data': {'error_code': 'no_face_detected'}
            }), 400

        embedding_obj = DeepFace.represent(
            img_path=temp_path,
            model_name=Config.FACE_RECOGNITION_MODEL,
            detector_backend=Config.FACE_DETECTION_MODEL,
            enforce_detection=True,
            align=True
        )

        embeddings = load_h5_embeddings()
        current_embedding = np.array(embedding_obj[0]['embedding'])
        matched_name, similarity = find_matching_face(current_embedding, embeddings)

        if not matched_name:
            response_data = {
                'status': 'error',
                'message': 'No matching face found in database',
                'data': {
                    'error_code': 'face_not_found',
                    'matched_name': None,
                    'confidence': None
                }
            }
            print(response_data)
            return jsonify(response_data), 404

        response_data = {
            'status': 'success',
            'message': 'Face recognition successful',
            'data': {
                'matched_name': matched_name,
                'confidence': float(similarity),
                'is_real': is_real,
                'spoof_confidence': spoof_confidence
            },
            'timing': {
                'total': f"{time.time() - start_time:.3f}s"
            },
            # 'location': {
            #     'location_type': location_type,
            #     'office_name': office_name,
            #     'latitude': latitude,
            #     'longitude': longitude,
            #     'timestamp': timestamp,
            # },
        }
        print(response_data)
        return jsonify(response_data), 200
    except Exception as e:
        response_data = {
            'status': 'error',
            'message': 'Face covvuld not be detected in face recognition.',
            'data': {'error_code': 'no_face_detected'}
        }
        print(response_data)
        return jsonify(response_data),  400
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")