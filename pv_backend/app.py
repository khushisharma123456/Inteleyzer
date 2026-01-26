"""
Main Flask application for the Pharmacovigilance Backend Service.
Post-Marketing Surveillance (PMS) / Pharmacovigilance system.
"""
import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pv_backend.config import config
from pv_backend.models import db


def create_app(config_name=None):
    """
    Application factory for creating Flask app instances.
    
    Args:
        config_name: Configuration to use ('development', 'production', 'testing')
    
    Returns:
        Configured Flask application
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions with proper CORS settings
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:5000", "http://127.0.0.1:5000", "*"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    db.init_app(app)
    
    # Register blueprints
    from pv_backend.auth.routes import auth_bp
    from pv_backend.routes.submission_routes import submission_bp
    from pv_backend.routes.case_routes import case_bp
    from pv_backend.routes.followup_routes import followup_bp
    from pv_backend.routes.excel_routes import excel_upload_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(submission_bp)
    app.register_blueprint(case_bp)
    app.register_blueprint(followup_bp)
    app.register_blueprint(excel_upload_bp)
    
    # Create tables
    with app.app_context():
        db.create_all()
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'pv-backend'}, 200
    
    # Root endpoint with API info
    @app.route('/')
    def index():
        return {
            'service': 'Pharmacovigilance Backend API',
            'version': '1.0.0',
            'endpoints': {
                'auth': '/api/auth',
                'submissions': '/api/submit',
                'cases': '/api/cases',
                'followups': '/api/followups',
                'excel': '/api/excel',
                'health': '/health'
            }
        }, 200
    
    return app


# Create app instance
app = create_app()


if __name__ == '__main__':
    app.run(debug=True, port=5001)
