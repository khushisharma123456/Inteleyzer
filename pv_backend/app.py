"""
Main Flask application for the Pharmacovigilance Backend Service.
Post-Marketing Surveillance (PMS) / Pharmacovigilance system.
"""
import os
from flask import Flask, render_template, redirect, url_for, session
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
    
    # Specify template and static folders pointing to backend folder
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'static'))
    
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config.from_object(config[config_name])
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'pv-secret-key-dev')
    
    # Initialize extensions with proper CORS settings
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:5001", "http://127.0.0.1:5001", "http://localhost:5000", "http://127.0.0.1:5000", "*"],
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
    from pv_backend.routes.pharmacy_report_routes import pharmacy_report_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(submission_bp)
    app.register_blueprint(case_bp)
    app.register_blueprint(followup_bp)
    app.register_blueprint(excel_upload_bp)
    app.register_blueprint(pharmacy_report_bp)
    
    # Create tables
    with app.app_context():
        db.create_all()
    
    # --- UI Routes ---
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/login')
    def login_page():
        return render_template('login.html')
    
    @app.route('/signup')
    def signup_page():
        return render_template('signup.html')
    
    # Doctor Routes
    @app.route('/doctor/dashboard')
    def doctor_dashboard():
        if 'user_id' not in session or session.get('role') != 'doctor':
            return redirect(url_for('login_page'))
        return render_template('doctor/dashboard.html')
    
    @app.route('/doctor/patients')
    def doctor_patients():
        if 'user_id' not in session or session.get('role') != 'doctor':
            return redirect(url_for('login_page'))
        return render_template('doctor/patients_v2.html')
    
    @app.route('/doctor/alerts')
    def doctor_alerts():
        if 'user_id' not in session or session.get('role') != 'doctor':
            return redirect(url_for('login_page'))
        return render_template('doctor/alerts.html')
    
    @app.route('/doctor/warnings')
    def doctor_warnings():
        if 'user_id' not in session or session.get('role') != 'doctor':
            return redirect(url_for('login_page'))
        return render_template('doctor/warnings.html')
    
    @app.route('/doctor/analysis')
    def doctor_analysis():
        if 'user_id' not in session or session.get('role') != 'doctor':
            return redirect(url_for('login_page'))
        return render_template('doctor/analysis.html')
    
    @app.route('/doctor/report')
    def doctor_report():
        if 'user_id' not in session or session.get('role') != 'doctor':
            return redirect(url_for('login_page'))
        return render_template('doctor/report.html')
    
    # Pharma Routes
    @app.route('/pharma/dashboard')
    def pharma_dashboard():
        if 'user_id' not in session or session.get('role') != 'pharma':
            return redirect(url_for('login_page'))
        return render_template('pharma/dashboard.html')
    
    @app.route('/pharma/reports')
    def pharma_reports():
        if 'user_id' not in session or session.get('role') != 'pharma':
            return redirect(url_for('login_page'))
        return render_template('pharma/reports.html')
    
    @app.route('/pharma/drugs')
    def pharma_drugs():
        if 'user_id' not in session or session.get('role') != 'pharma':
            return redirect(url_for('login_page'))
        return render_template('pharma/drugs.html')
    
    @app.route('/pharma/analysis')
    def pharma_analysis():
        if 'user_id' not in session or session.get('role') != 'pharma':
            return redirect(url_for('login_page'))
        return render_template('pharma/analysis.html')
    
    # Pharmacy Routes
    @app.route('/pharmacy/dashboard')
    def pharmacy_dashboard():
        if 'user_id' not in session or session.get('role') != 'pharmacy':
            return redirect(url_for('login_page'))
        return render_template('pharmacy/dashboard.html')
    
    @app.route('/pharmacy/alerts')
    def pharmacy_alerts():
        if 'user_id' not in session or session.get('role') != 'pharmacy':
            return redirect(url_for('login_page'))
        return render_template('pharmacy/alerts.html')
    
    @app.route('/pharmacy/report')
    def pharmacy_report():
        if 'user_id' not in session or session.get('role') != 'pharmacy':
            return redirect(url_for('login_page'))
        return render_template('pharmacy/report.html')
    
    @app.route('/pharmacy/reports')
    def pharmacy_reports():
        if 'user_id' not in session or session.get('role') != 'pharmacy':
            return redirect(url_for('login_page'))
        return render_template('pharmacy/reports.html')
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'pv-backend'}, 200
    
    return app


# Create app instance
app = create_app()
