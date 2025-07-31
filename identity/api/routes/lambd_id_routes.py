# ═══════════════════════════════════════════════════════════════════════════
# FILENAME: lambd_id_routes.py
# MODULE: lukhas_id.api.routes.lambd_id_routes
# DESCRIPTION: Defines Flask Blueprint and API routes for LUKHAS ΛiD (Lambda ID)
#              generation, validation, entropy calculation, and tier management.
# DEPENDENCIES: Flask, Flask-Limiter, logging, typing, LambdaIDController,
#               other core LUKHAS ID services (indirectly via controller).
# LICENSE: PROPRIETARY - LUKHAS AI SYSTEMS - UNAUTHORIZED ACCESS PROHIBITED
# ═══════════════════════════════════════════════════════════════════════════

import logging
import time # For request IDs
from flask import Blueprint, request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from typing import Dict, Any, Optional # Added Optional

# Initialize ΛTRACE logger for this routes module
logger = logging.getLogger("ΛTRACE.lukhas_id.api.routes.lambd_id")
logger.info("ΛTRACE: Initializing lambd_id_routes module.")

# Attempt to import controller and core services (though controller should handle service imports)
try:
    from ..controllers.lambd_id_controller import LambdaIDController
    # These specific imports might be redundant if controller handles them, but kept for context
    # from ...core.id_service.lambd_id_generator import LambdaIDGenerator
    # from ...core.id_service.lambd_id_validator import LambdaIDValidator
    # from ...core.id_service.lambd_id_entropy import EntropyCalculator
    logger.info("ΛTRACE: LambdaIDController imported successfully.")
except ImportError as e:
    logger.error(f"ΛTRACE: Failed to import LambdaIDController or core services: {e}. ΛiD routes may not function.", exc_info=True)
    # Define a fallback controller if import fails, to allow app to load but endpoints will fail
    class LambdaIDController: # type: ignore
        def __init__(self): logger.error("ΛTRACE: Using FALLBACK LambdaIDController.")
        def generate_id(self, **kwargs): return {'success': False, 'error': 'Controller not loaded'}
        def validate_id(self, **kwargs): return {'success': False, 'error': 'Controller not loaded'}
        def calculate_entropy(self, **kwargs): return {'success': False, 'error': 'Controller not loaded'}
        def get_tier_information(self, **kwargs): return {'success': False, 'error': 'Controller not loaded'}
        def request_tier_upgrade(self, **kwargs): return {'success': False, 'error': 'Controller not loaded'}
        def check_service_health(self, **kwargs): return {'all_services_up': False, 'error': 'Controller not loaded'}

# Initialize Flask Blueprint for LambdaID routes
# Using a more specific versioning in URL prefix if this is v1 of these specific routes
lambd_id_bp = Blueprint('lambd_id_v1', __name__, url_prefix='/api/v1/lambda-id')
logger.info(f"ΛTRACE: Flask Blueprint 'lambd_id_v1' created with url_prefix: {lambd_id_bp.url_prefix}")

# Initialize rate limiter.
# For this to work effectively, the Limiter instance needs to be initialized with the Flask app.
# This is often done in the app factory (e.g., in lukhas_id.api.__init__.py).
# If `limiter` is defined here globally and `init_app` is not called on it with the main app,
# its decorators might not function as expected unless Flask-Limiter handles this context.
# For now, assuming it's either configured with an app elsewhere or works standalone on blueprint routes.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per hour", "50 per minute"] # Adjusted default limits
)
logger.info(f"ΛTRACE: Flask-Limiter initialized for lambd_id_bp with default limits.")
# It's common to call limiter.init_app(app) in the app factory.
# If these routes are part of `app` from `unified_api.py` which has its own limiter,
# this blueprint-specific limiter might conflict or be redundant.
# Assuming this blueprint's limiter is intended.

# Initialize the LambdaIDController
try:
    controller = LambdaIDController()
    logger.info("ΛTRACE: LambdaIDController instance created.")
except Exception as e_controller:
    logger.error(f"ΛTRACE: Failed to instantiate LambdaIDController: {e_controller}. Endpoints will likely fail.", exc_info=True)
    controller = LambdaIDController() # Fallback instance

# Helper for request IDs
def _get_req_id(prefix="req"):
    return f"{prefix}_{int(time.time()*1000)}_{random.randint(100,999)}"

# --- API Route Definitions ---

# Human-readable comment: Endpoint to generate a new Lambda ID.
@lambd_id_bp.route('/generate', methods=['POST'])
@limiter.limit("10 per minute; 200 per hour") # More specific limit for this route
def generate_lambda_id_route(): # Renamed for clarity
    """
    Generates a new LUKHAS ΛiD based on user tier, symbolic preferences,
    and other optional configurations.
    """
    req_id = _get_req_id("gen_lid")
    logger.info(f"ΛTRACE ({req_id}): POST /generate request received from {request.remote_addr}.")
    try:
        if not request.is_json:
            logger.warning(f"ΛTRACE ({req_id}): Invalid Content-Type for /generate. Expected application/json.")
            return jsonify({'success': False, 'error': 'Content-Type must be application/json', 'error_code': 'INVALID_CONTENT_TYPE'}), 400

        request_data = request.get_json()
        logger.debug(f"ΛTRACE ({req_id}): Request data for /generate: {request_data}")

        user_tier = request_data.get('user_tier', 0) # Default to tier 0
        # Ensure all parameters are correctly passed to the controller
        result = controller.generate_id(
            user_tier=int(user_tier), # Ensure type
            symbolic_preferences=request_data.get('symbolic_preferences', []),
            entropy_requirements=request_data.get('entropy_requirements', {}),
            commercial_options=request_data.get('commercial_options', {}),
            request_metadata={ # Pass along request context for detailed logging by controller
                'ip_address': get_remote_address(), 'user_agent': request.headers.get('User-Agent'),
                'request_timestamp': datetime.now().isoformat(), 'endpoint': request.path
            }
        )

        status_code = 201 if result.get('success') else 400 # 201 Created for success
        log_level = logger.info if status_code == 201 else logger.warning
        log_level(f"ΛTRACE ({req_id}): /generate response. Success: {result.get('success')}, LambdaID: {result.get('lambda_id', 'N/A')}, Error: {result.get('error')}, Status Code: {status_code}")
        return jsonify(result), status_code

    except Exception as e:
        logger.error(f"ΛTRACE ({req_id}): Unhandled error in /generate endpoint: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error during ID generation.', 'error_code': 'INTERNAL_GENERATION_ERROR'}), 500

# Human-readable comment: Endpoint to validate an existing Lambda ID.
@lambd_id_bp.route('/validate', methods=['POST'])
@limiter.limit("50 per minute; 1000 per hour")
def validate_lambda_id_route(): # Renamed
    """
    Validates an existing LUKHAS ΛiD against format, entropy, tier compliance,
    and optionally checks for collisions.
    """
    req_id = _get_req_id("val_lid")
    logger.info(f"ΛTRACE ({req_id}): POST /validate request received from {request.remote_addr}.")
    try:
        if not request.is_json:
            logger.warning(f"ΛTRACE ({req_id}): Invalid Content-Type for /validate.")
            return jsonify({'success': False, 'error': 'Content-Type must be application/json', 'error_code': 'INVALID_CONTENT_TYPE'}), 400

        request_data = request.get_json()
        logger.debug(f"ΛTRACE ({req_id}): Request data for /validate: {request_data}")
        lambda_id_to_validate = request_data.get('lambda_id')

        if not lambda_id_to_validate:
            logger.warning(f"ΛTRACE ({req_id}): Missing 'lambda_id' in /validate request.")
            return jsonify({'success': False, 'error': 'lambda_id is required in request body.', 'error_code': 'MISSING_LAMBDA_ID'}), 400

        result = controller.validate_id(
            lambda_id=lambda_id_to_validate,
            validation_level=request_data.get('validation_level', 'standard'),
            check_collision=request_data.get('check_collision', False),
            request_metadata={'ip_address': get_remote_address(), 'endpoint': request.path}
        )

        logger.info(f"ΛTRACE ({req_id}): /validate response for ΛiD '{lambda_id_to_validate}'. Valid: {result.get('valid', False)}, Details: {result.get('validation_details')}")
        return jsonify(result), 200 # Validation itself succeeded or failed, but API call is 200

    except Exception as e:
        logger.error(f"ΛTRACE ({req_id}): Unhandled error in /validate endpoint: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error during ID validation.', 'error_code': 'INTERNAL_VALIDATION_ERROR'}), 500

# Human-readable comment: Endpoint to calculate entropy for symbolic input.
@lambd_id_bp.route('/entropy', methods=['POST'])
@limiter.limit("30 per minute; 500 per hour")
def calculate_entropy_route(): # Renamed
    """
    Calculates the entropy score for a given list of symbolic inputs,
    optionally considering user tier and calculation method.
    """
    req_id = _get_req_id("entropy")
    logger.info(f"ΛTRACE ({req_id}): POST /entropy request received from {request.remote_addr}.")
    try:
        if not request.is_json:
            logger.warning(f"ΛTRACE ({req_id}): Invalid Content-Type for /entropy.")
            return jsonify({'success': False, 'error': 'Content-Type must be application/json', 'error_code': 'INVALID_CONTENT_TYPE'}), 400

        request_data = request.get_json()
        logger.debug(f"ΛTRACE ({req_id}): Request data for /entropy: {request_data}")
        symbolic_input_list = request_data.get('symbolic_input', [])

        if not symbolic_input_list:
            logger.warning(f"ΛTRACE ({req_id}): Missing 'symbolic_input' in /entropy request.")
            return jsonify({'success': False, 'error': 'symbolic_input (list of strings) is required.', 'error_code': 'MISSING_SYMBOLIC_INPUT'}), 400

        result = controller.calculate_entropy(
            symbolic_input=symbolic_input_list,
            tier=int(request_data.get('tier', 0)),
            calculation_method=request_data.get('calculation_method', 'shannon'),
            request_metadata={'ip_address': get_remote_address(), 'endpoint': request.path}
        )

        logger.info(f"ΛTRACE ({req_id}): /entropy response. Score: {result.get('entropy_score', 'N/A')}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"ΛTRACE ({req_id}): Unhandled error in /entropy endpoint: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error during entropy calculation.', 'error_code': 'INTERNAL_ENTROPY_ERROR'}), 500

# Human-readable comment: Endpoint to get tier information.
@lambd_id_bp.route('/tiers', methods=['GET'])
@limiter.limit("100 per hour") # Less frequent, allow more
def get_tier_information_route(): # Renamed
    """
    Retrieves information about LUKHAS ΛiD tiers, either for a specific tier
    or all tiers, optionally including progression maps.
    """
    req_id = _get_req_id("tiers_info")
    logger.info(f"ΛTRACE ({req_id}): GET /tiers request received from {request.remote_addr}. Args: {request.args}")
    try:
        specific_tier_req = request.args.get('tier', default=None, type=int)
        include_prog_req = request.args.get('include_progression', default='false', type=str).lower() == 'true'

        result = controller.get_tier_information(
            specific_tier=specific_tier_req,
            include_progression=include_prog_req,
            request_metadata={'ip_address': get_remote_address(), 'endpoint': request.path}
        )

        logger.info(f"ΛTRACE ({req_id}): /tiers response. Success: {result.get('success')}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"ΛTRACE ({req_id}): Unhandled error in /tiers endpoint: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error retrieving tier information.', 'error_code': 'INTERNAL_TIER_INFO_ERROR'}), 500

# Human-readable comment: Endpoint to request a tier upgrade.
@lambd_id_bp.route('/upgrade', methods=['POST'])
@limiter.limit("5 per hour; 1 per minute") # Stricter limit for upgrades
def request_tier_upgrade_route(): # Renamed
    """
    Handles a request to upgrade a user's LUKHAS ΛiD to a target tier.
    Requires validation data and current ΛiD.
    """
    req_id = _get_req_id("upgrade_tier")
    logger.info(f"ΛTRACE ({req_id}): POST /upgrade request received from {request.remote_addr}.")
    try:
        if not request.is_json:
            logger.warning(f"ΛTRACE ({req_id}): Invalid Content-Type for /upgrade.")
            return jsonify({'success': False, 'error': 'Content-Type must be application/json', 'error_code': 'INVALID_CONTENT_TYPE'}), 400

        request_data = request.get_json()
        logger.debug(f"ΛTRACE ({req_id}): Request data for /upgrade: {request_data}")
        current_lambda_id_val = request_data.get('current_lambda_id')
        target_tier_val = request_data.get('target_tier') # Should be int

        if not current_lambda_id_val or target_tier_val is None: # Check for None explicitly for target_tier=0
            logger.warning(f"ΛTRACE ({req_id}): Missing 'current_lambda_id' or 'target_tier' in /upgrade request.")
            return jsonify({'success': False, 'error': "'current_lambda_id' and 'target_tier' are required.", 'error_code': 'MISSING_UPGRADE_PARAMS'}), 400

        result = controller.request_tier_upgrade(
            current_lambda_id=current_lambda_id_val,
            target_tier=int(target_tier_val),
            validation_data=request_data.get('validation_data', {}),
            request_metadata={'ip_address': get_remote_address(), 'user_agent': request.headers.get('User-Agent'), 'endpoint': request.path}
        )

        status_code = 200 if result.get('success') and result.get('upgrade_approved') else (202 if result.get('success') else 400) # 202 Accepted if not approved but request ok
        log_level = logger.info if status_code < 400 else logger.warning
        log_level(f"ΛTRACE ({req_id}): /upgrade response. Approved: {result.get('upgrade_approved')}, New ΛiD: {result.get('new_lambda_id', 'N/A')}, Error: {result.get('error')}, Status Code: {status_code}")
        return jsonify(result), status_code

    except Exception as e:
        logger.error(f"ΛTRACE ({req_id}): Unhandled error in /upgrade endpoint: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error during tier upgrade.', 'error_code': 'INTERNAL_UPGRADE_ERROR'}), 500

# Human-readable comment: Health check endpoint for this blueprint/service.
@lambd_id_bp.route('/health', methods=['GET'])
def health_check_route(): # Renamed
    """Provides a health check for the LambdaID service, including its controller dependencies."""
    req_id = _get_req_id("health_lid")
    logger.info(f"ΛTRACE ({req_id}): GET /health request received from {request.remote_addr}.")
    try:
        health_status_details = controller.check_service_health(
            request_metadata={'ip_address': get_remote_address(), 'endpoint': request.path}
        ) # Controller method logs its details

        response_payload = {
            'status': 'healthy' if health_status_details.get('overall_status') == 'healthy' else 'degraded',
            'version': health_status_details.get('controller_version', 'N/A'), # Assuming controller provides its version
            'timestamp': health_status_details.get('timestamp', datetime.now().isoformat()),
            'services_checked': health_status_details.get('service_details', {})
        }
        http_status = 200 if response_payload['status'] == 'healthy' else 503 # 503 Service Unavailable if degraded
        logger.info(f"ΛTRACE ({req_id}): /health response. Overall Status: {response_payload['status']}, HTTP Status: {http_status}")
        return jsonify(response_payload), http_status

    except Exception as e:
        logger.error(f"ΛTRACE ({req_id}): Unhandled error in /health endpoint: {e}", exc_info=True)
        return jsonify({'status': 'unhealthy', 'error_details': str(e), 'timestamp': datetime.now().isoformat()}), 503

# --- Blueprint Error Handlers ---
# Human-readable comment: Custom error handler for rate limit exceeded on this blueprint.
@lambd_id_bp.errorhandler(429) # Assuming limiter is attached to this blueprint or app
def handle_rate_limit_exceeded_on_bp(e): # Renamed
    """Handles 429 Rate Limit Exceeded errors specifically for the lambd_id_bp."""
    req_id = _get_req_id("err429")
    # The error 'e' from Flask-Limiter might have a description attribute.
    error_description = getattr(e, 'description', 'Rate limit exceeded for this endpoint.')
    logger.warning(f"ΛTRACE ({req_id}): Rate limit exceeded for {request.remote_addr} on {request.path}. Description: {error_description}")
    return jsonify({'success': False, 'error': 'Rate limit exceeded', 'error_code': 'RATE_LIMIT_EXCEEDED_LIDBP', 'message': error_description}), 429

# Human-readable comment: Custom error handler for 404 on this blueprint.
@lambd_id_bp.errorhandler(404) # This handles 404s *within* this blueprint's prefix
def handle_not_found_on_bp(e): # Renamed
    """Handles 404 Not Found errors for routes under lambd_id_bp."""
    req_id = _get_req_id("err404")
    logger.warning(f"ΛTRACE ({req_id}): Resource not found within lambd_id_bp at {request.path}.")
    return jsonify({'success': False, 'error': 'Endpoint not found within LambdaID service.', 'error_code': 'LIDBP_NOT_FOUND'}), 404

# Human-readable comment: Custom error handler for 405 on this blueprint.
@lambd_id_bp.errorhandler(405) # Method Not Allowed within this blueprint
def handle_method_not_allowed_on_bp(e): # Renamed
    """Handles 405 Method Not Allowed errors for routes under lambd_id_bp."""
    req_id = _get_req_id("err405")
    logger.warning(f"ΛTRACE ({req_id}): Method {request.method} not allowed for {request.path} within lambd_id_bp.")
    return jsonify({'success': False, 'error': 'Method not allowed for this LambdaID resource.', 'error_code': 'LIDBP_METHOD_NOT_ALLOWED'}), 405

logger.info("ΛTRACE: lambd_id_routes module fully loaded and configured.")

# ═══════════════════════════════════════════════════════════════════════════
# FILENAME: lambd_id_routes.py
# VERSION: 1.1.0 # Updated version
# TIER SYSTEM: Tier requirements are typically enforced by controller or service layer, not directly in routes.
# ΛTRACE INTEGRATION: ENABLED
# CAPABILITIES: Defines API routes for ΛiD generation, validation, entropy calculation,
#               tier information, tier upgrades, and health checks.
# FUNCTIONS: generate_lambda_id_route, validate_lambda_id_route, calculate_entropy_route,
#            get_tier_information_route, request_tier_upgrade_route, health_check_route,
#            handle_rate_limit_exceeded_on_bp, handle_not_found_on_bp, handle_method_not_allowed_on_bp,
#            _get_req_id.
# CLASSES: None.
# DECORATORS: @lambd_id_bp.route, @limiter.limit, @lambd_id_bp.errorhandler.
# DEPENDENCIES: Flask (Blueprint, request, jsonify, current_app), Flask-Limiter, logging, typing, time,
#               ..controllers.lambd_id_controller.LambdaIDController.
# INTERFACES: Exposes HTTP endpoints under the /api/v1/lambda-id prefix.
# ERROR HANDLING: Returns JSON error responses for client errors (4xx) and server errors (5xx).
#                 Custom error handlers for 404, 405, 429 specific to this blueprint.
# LOGGING: ΛTRACE_ENABLED for all route handling, including request details, controller calls,
#          responses, and errors.
# AUTHENTICATION: Assumed to be handled by a global mechanism or middleware if routes require auth.
#                 (No specific @require_auth decorator seen here, unlike some other API files).
# HOW TO USE:
#   Register `lambd_id_bp` with the main Flask application instance.
#   e.g., app.register_blueprint(lambd_id_bp)
# INTEGRATION NOTES: Relies on LambdaIDController for business logic.
#                    Rate limiting is applied per route. Ensure Limiter is properly initialized with the app.
#                    Consider standardizing error response structure further with `api_response` if used globally.
# MAINTENANCE: Update routes and request/response models as controller capabilities evolve.
#              Monitor rate limits and adjust as needed.
# CONTACT: LUKHAS DEVELOPMENT TEAM
# LICENSE: PROPRIETARY - LUKHAS AI SYSTEMS - UNAUTHORIZED ACCESS PROHIBITED
# ═══════════════════════════════════════════════════════════════════════════
