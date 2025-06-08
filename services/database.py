import psycopg2
from psycopg2.extras import DictCursor
from config.settings import Config
from datetime import datetime

def get_db_connection():
    try:
        connection = psycopg2.connect(**Config.DB_CONFIG)
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def save_user(username, email, role, password="123"):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                INSERT INTO karyawan (nama_karyawan, email, role, password) 
                VALUES (%s, %s, %s, %s)
                RETURNING id_karyawan
            """
            cursor.execute(sql, (username, email, role, password))
            connection.commit()
            return {'message': 'User saved successfully', 'password': password}, 201
    except Exception as e:
        return {'error': str(e)}, 500
    finally:
        connection.close()

def get_user_by_credentials(email, password):
    connection = get_db_connection()
    try:
        with connection.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute("""
                SELECT id_karyawan, email, password, nama_karyawan, role
                FROM karyawan 
                WHERE email = %s AND password = %s
            """, (email, password))
            return cursor.fetchone()
    finally:
        connection.close()

def save_presence(data):
    connection = get_db_connection()
    try:
        with connection.cursor(cursor_factory=DictCursor) as cursor:
            sql = """
                INSERT INTO absensi 
                (id_karyawan, nama_karyawan, work_type, office, latitude, longitude, absensi_masuk) 
                VALUES (
                    (SELECT id_karyawan FROM karyawan WHERE nama_karyawan = %s), 
                    %s, %s, %s, %s, %s, %s
                )
            """
            cursor.execute(sql, (
                data['database_name'],
                data['database_name'],
                data['location_type'],
                data['office_name'],
                data['latitude'],
                data['longitude'],
                data['timestamp']
            ))
            connection.commit()
            return {'status': 'success', 'message': 'Presence recorded successfully'}, 201
    except Exception as e:
        print(f"Error saving presence: {e}")
        return {'error': str(e)}, 500
    finally:
        connection.close()

def check_presence(user_name=None, user_id=None, check_type='in'):
    connection = get_db_connection()
    try:
        with connection.cursor(cursor_factory=DictCursor) as cursor:
            if check_type == 'in':
                date_field = 'absensi_masuk'
            else:
                date_field = 'absensi_pulang'
                
            if user_id:
                sql = f"""
                    SELECT * FROM absensi 
                    WHERE id_karyawan = %s AND DATE({date_field}) = CURRENT_DATE
                """
                cursor.execute(sql, (user_id,))
            else:
                sql = f"""
                    SELECT * FROM absensi 
                    WHERE nama_karyawan = %s AND DATE({date_field}) = CURRENT_DATE
                """
                cursor.execute(sql, (user_name,))
                
            result = cursor.fetchone()
            if result:
                return {
                    'status': 'success',
                    'message': f"User telah melakukan presensi {'masuk' if check_type == 'in' else 'pulang'} hari ini.",
                    'data': dict(result)
                }, 200
            else:
                return {
                    'status': 'not_found',
                    'message': f"User belum melakukan presensi {'masuk' if check_type == 'in' else 'pulang'} hari ini."
                }, 404
    except Exception as e:
        return {'status': 'error', 'message': f'Error: {str(e)}'}, 500
    finally:
        connection.close()

def get_history(user_name):
    connection = get_db_connection()
    try:
        with connection.cursor(cursor_factory=DictCursor) as cursor:
            sql = """
                SELECT 
                    TO_CHAR(absensi_masuk, 'YYYY-MM-DD') as tanggal,
                    TO_CHAR(absensi_masuk, 'HH24:MI:SS') as jam,
                    nama_karyawan,
                    work_type,
                    office
                FROM absensi 
                WHERE nama_karyawan = %s
                ORDER BY absensi_masuk DESC
            """
            cursor.execute(sql, (user_name,))
            records = cursor.fetchall()
            if not records:
                return {'status': 'error', 'message': 'Data absensi tidak ditemukan'}, 404
            return {
                'status': 'success',
                'data': [{
                    'tanggal': record['tanggal'],
                    'jam': record['jam'],
                    'nama_karyawan': record['nama_karyawan'],
                    'work_type': record['work_type'],
                    'office': record['office']
                } for record in records]
            }, 200
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500
    finally:
        connection.close()

def get_employees():
    connection = get_db_connection()
    try:
        with connection.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute("SELECT id_karyawan, nama_karyawan, email, password, role FROM karyawan")
            employees = cursor.fetchall()
            return [dict(emp) for emp in employees], 200
    except Exception as e:
        print(f"Error fetching employees: {e}")
        return {"error": "Error fetching employees"}, 500
    finally:
        connection.close()

def delete_employee(id_karyawan):
    connection = get_db_connection()
    try:
        with connection.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute("SELECT nama_karyawan FROM karyawan WHERE id_karyawan = %s", (id_karyawan,))
            result = cursor.fetchone()
            if not result:
                return {"error": "Employee not found"}, 404
            nama_karyawan = result['nama_karyawan']
            cursor.execute("DELETE FROM karyawan WHERE id_karyawan = %s", (id_karyawan,))
            connection.commit()
            return {"message": "Employee deleted successfully", "nama_karyawan": nama_karyawan}, 200
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        connection.close()

def get_employee_by_id(employee_id):
    connection = get_db_connection()
    try:
        with connection.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute("SELECT * FROM karyawan WHERE id_karyawan = %s", (employee_id,))
            employee = cursor.fetchone()
            if employee:
                return dict(employee), 200
            else:
                return {"error": "Employee not found"}, 404
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        connection.close()