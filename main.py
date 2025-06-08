from flask import Flask
from flask_cors import CORS
from routes.api_route import api_route_bp
from services.auth import init_jwt
from config.settings import Config
from dotenv import load_dotenv
import os



# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Apply configuration from settings
app.config['SECRET_KEY'] = Config.JWT_SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # Default to 16MB
CORS(app)

# Initialize JWT
init_jwt(app)

# Register blueprint
app.register_blueprint(api_route_bp)

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST')
    port = int(os.getenv('FLASK_PORT'))
    debug = os.getenv('DEBUG').lower() == 'true'
    app.run(debug=debug, host=host, port=port)