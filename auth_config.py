"""
JWT Authentication Configuration
Handles token generation, validation, and session management
"""
import jwt
import os
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, session

# JWT Configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'

# Session Configuration
SESSION_TIMEOUT_MINUTES = 30  # Session expires after 30 minutes of inactivity
TOKEN_EXPIRY_HOURS = 24  # JWT token expires after 24 hours

class JWTConfig:
    """JWT Configuration and Token Management"""
    
    @staticmethod
    def generate_token(user_id, email, role, name):
        """
        Generate JWT token for user
        
        Args:
            user_id: User ID
            email: User email
            role: User role (doctor, pharma, pharmacy, hospital)
            name: User name
            
        Returns:
            JWT token string
        """
        payload = {
            'user_id': user_id,
            'email': email,
            'role': role,
            'name': name,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)
        }
        
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return token
    
    @staticmethod
    def verify_token(token):
        """
        Verify JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded payload if valid, None if invalid
        """
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None  # Token expired
        except jwt.InvalidTokenError:
            return None  # Invalid token
    
    @staticmethod
    def get_token_from_request():
        """
        Extract JWT token from request headers
        
        Returns:
            Token string or None
        """
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            return auth_header[7:]  # Remove 'Bearer ' prefix
        return None


def token_required(f):
    """
    Decorator to require valid JWT token
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = JWTConfig.get_token_from_request()
        
        if not token:
            return jsonify({'success': False, 'message': 'Token missing'}), 401
        
        payload = JWTConfig.verify_token(token)
        
        if not payload:
            return jsonify({'success': False, 'message': 'Token expired or invalid'}), 401
        
        # Store user info in request context
        request.user_id = payload['user_id']
        request.user_email = payload['email']
        request.user_role = payload['role']
        request.user_name = payload['name']
        
        return f(*args, **kwargs)
    
    return decorated


def session_required(f):
    """
    Decorator to require valid session with expiry check
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        # Check session expiry
        if 'session_start' in session:
            session_age = datetime.utcnow() - session['session_start']
            if session_age > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                session.clear()
                return jsonify({'success': False, 'message': 'Session expired'}), 401
        
        # Update session timestamp (refresh session)
        session['session_start'] = datetime.utcnow()
        
        return f(*args, **kwargs)
    
    return decorated
