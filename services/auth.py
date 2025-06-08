from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity
from config.settings import Config

jwt = JWTManager()

def init_jwt(app):
    jwt.init_app(app)
    app.config['JWT_SECRET_KEY'] = Config.JWT_SECRET_KEY
    # app.config['JWT_ACCESS_TOKEN_EXPIRES'] = Config.JWT_ACCESS_TOKEN_EXPIRES
    app.config['JWT_TOKEN_LOCATION'] = Config.JWT_TOKEN_LOCATION
    app.config['JWT_HEADER_NAME'] = Config.JWT_HEADER_NAME
    app.config['JWT_HEADER_TYPE'] = Config.JWT_HEADER_TYPE

def get_current_user():
    try:
        return get_jwt_identity()
    except Exception:
        return None