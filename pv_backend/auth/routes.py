"""
Authentication API routes.
Handles user registration, login, token refresh, and pharmacy verification.
"""
from flask import Blueprint, request, jsonify, g, session
from datetime import datetime

from pv_backend.models import db, User, UserRole, AuditLog, AuditAction
from pv_backend.auth import (
    generate_token, decode_token, token_required, 
    admin_required, roles_required
)

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')


@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Register a new user.
    
    PV Context:
    - All user registrations are audited
    - Pharmacies start as unverified
    - Role is required and validated
    """
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['email', 'password', 'full_name', 'role']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({
            'success': False,
            'message': f'Missing required fields: {missing}'
        }), 400
    
    # Validate role
    try:
        role = UserRole(data['role'])
    except ValueError:
        return jsonify({
            'success': False,
            'message': f'Invalid role. Must be one of: {[r.value for r in UserRole]}'
        }), 400
    
    # Check if email already exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({
            'success': False,
            'message': 'Email already registered'
        }), 409
    
    # Create user
    user = User(
        email=data['email'],
        full_name=data['full_name'],
        role=role,
        organization=data.get('organization'),
        license_number=data.get('license_number'),
        is_verified=False if role == UserRole.PHARMACY else True  # Pharmacies need verification
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    
    # Audit log
    AuditLog.log(
        action=AuditAction.CREATE,
        user=None,  # No user context yet
        entity_type='User',
        entity_id=None,  # Will be set after commit
        entity_identifier=user.email,
        state_after=user.to_dict(),
        reason='User registration',
        request=request
    )
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'User registered successfully',
        'user': user.to_dict(),
        'requires_verification': role == UserRole.PHARMACY
    }), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Authenticate user and return JWT tokens.
    
    PV Context:
    - Login attempts are audited
    - Returns both access and refresh tokens
    - Pharmacy verification status is included in response
    """
    data = request.get_json()
    
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({
            'success': False,
            'message': 'Email and password required'
        }), 400
    
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not user.check_password(data['password']):
        # Audit failed login attempt
        AuditLog.log(
            action=AuditAction.LOGIN,
            user=None,
            entity_type='User',
            entity_identifier=data.get('email'),
            state_after={'success': False},
            reason='Failed login attempt',
            request=request
        )
        db.session.commit()
        
        return jsonify({
            'success': False,
            'message': 'Invalid email or password'
        }), 401
    
    if not user.is_active or user.is_deleted:
        return jsonify({
            'success': False,
            'message': 'Account is inactive or deleted'
        }), 401
    
    # Update last login
    user.last_login_at = datetime.utcnow()
    
    # Set Flask session for HTML route protection
    session['user_id'] = user.id
    session['role'] = user.role.value
    session['user_name'] = user.full_name
    
    # Generate tokens
    access_token = generate_token(user, 'access')
    refresh_token = generate_token(user, 'refresh')
    
    # Audit successful login
    AuditLog.log(
        action=AuditAction.LOGIN,
        user=user,
        entity_type='User',
        entity_id=user.id,
        entity_identifier=user.email,
        state_after={'success': True},
        reason='Successful login',
        request=request
    )
    
    db.session.commit()
    
    response = {
        'success': True,
        'message': 'Login successful',
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': user.to_dict()
    }
    
    # Warn unverified pharmacies
    if user.role == UserRole.PHARMACY and not user.is_verified:
        response['warning'] = 'Pharmacy not yet verified. You cannot submit data until verified.'
    
    return jsonify(response), 200


@auth_bp.route('/refresh', methods=['POST'])
def refresh_token():
    """
    Refresh access token using refresh token.
    """
    data = request.get_json()
    
    if not data or not data.get('refresh_token'):
        return jsonify({
            'success': False,
            'message': 'Refresh token required'
        }), 400
    
    payload = decode_token(data['refresh_token'])
    if not payload or payload.get('token_type') != 'refresh':
        return jsonify({
            'success': False,
            'message': 'Invalid refresh token'
        }), 401
    
    user = User.query.get(payload['user_id'])
    if not user or user.is_deleted or not user.is_active:
        return jsonify({
            'success': False,
            'message': 'User account is inactive'
        }), 401
    
    new_access_token = generate_token(user, 'access')
    
    return jsonify({
        'success': True,
        'access_token': new_access_token
    }), 200


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """
    Logout user by clearing session.
    """
    session.clear()
    return jsonify({
        'success': True,
        'message': 'Logged out successfully'
    }), 200


@auth_bp.route('/me', methods=['GET'])
@token_required
def get_current_user():
    """
    Get current authenticated user's information.
    """
    user = g.current_user
    return jsonify({
        'success': True,
        'user': user.to_dict()
    }), 200


@auth_bp.route('/verify-pharmacy/<int:user_id>', methods=['POST'])
@token_required
@admin_required
def verify_pharmacy(user_id):
    """
    Verify a pharmacy user (Admin only).
    
    PV Context:
    - Only admins can verify pharmacies
    - Verification is required before pharmacies can submit data
    - Full audit trail is maintained
    """
    pharmacy_user = User.query.get(user_id)
    
    if not pharmacy_user:
        return jsonify({
            'success': False,
            'message': 'User not found'
        }), 404
    
    if pharmacy_user.role != UserRole.PHARMACY:
        return jsonify({
            'success': False,
            'message': 'User is not a pharmacy'
        }), 400
    
    if pharmacy_user.is_verified:
        return jsonify({
            'success': False,
            'message': 'Pharmacy is already verified'
        }), 400
    
    # Store before state
    state_before = pharmacy_user.to_dict()
    
    # Verify pharmacy
    admin = g.current_user
    pharmacy_user.is_verified = True
    pharmacy_user.verified_at = datetime.utcnow()
    pharmacy_user.verified_by_id = admin.id
    
    # Audit log
    AuditLog.log(
        action=AuditAction.UPDATE,
        user=admin,
        entity_type='User',
        entity_id=pharmacy_user.id,
        entity_identifier=pharmacy_user.email,
        state_before=state_before,
        state_after=pharmacy_user.to_dict(),
        changes={'is_verified': True, 'verified_by': admin.email},
        reason='Pharmacy verification by admin',
        request=request
    )
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Pharmacy verified successfully',
        'user': pharmacy_user.to_dict()
    }), 200


@auth_bp.route('/unverified-pharmacies', methods=['GET'])
@token_required
@admin_required
def get_unverified_pharmacies():
    """
    Get list of unverified pharmacies (Admin only).
    """
    pharmacies = User.query.filter_by(
        role=UserRole.PHARMACY,
        is_verified=False,
        is_deleted=False
    ).all()
    
    return jsonify({
        'success': True,
        'pharmacies': [p.to_dict() for p in pharmacies]
    }), 200
