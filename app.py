"""
Main entry point for the Pharmacovigilance Backend Application
Run this file to start the Flask server
"""
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pv_backend.app import app

if __name__ == '__main__':
    # Run the Flask app on port 5001
    # Access at http://localhost:5001
    app.run(debug=True, host='127.0.0.1', port=5001)
