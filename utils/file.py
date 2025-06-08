import os
from werkzeug.utils import secure_filename

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def get_incremental_filename(folder, filename):
    """Generate an incremental filename to avoid overwriting existing files."""
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = f"{name}_{counter}{ext}"
    new_file_path = os.path.join(folder, new_filename)

    while os.path.exists(new_file_path):
        counter += 1
        new_filename = f"{name}_{counter}{ext}"
        new_file_path = os.path.join(folder, new_filename)
        
    return new_filename