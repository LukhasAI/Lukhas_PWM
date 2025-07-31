"""
Mobile module alias for lukhas-id mobile platform functionality.
Points to mobile_platform directory for backward compatibility.

ΛTAG: mobile_platform, lukhas_id, compatibility
"""

# Re-export mobile_platform components for backward compatibility
# Explicit imports replacing star imports per PEP8 guidelines # CLAUDE_EDIT_v0.8
from mobile_platform.mobile_ui_renderer import (
    TouchGesture, VisualizationMode, TouchEvent
)


# Create placeholder classes for missing components
class MobileWebSocketClient:
    """Placeholder for mobile WebSocket client functionality."""

    def __init__(self):
        self.connected = False

    def connect(self):
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False
        return True

    def send_message(self, message):
        return {"status": "mock_sent", "message": message}


class QRCodeAnimator:
    """Placeholder for QR code animation functionality."""

    def __init__(self):
        self.animation_active = False

    def start_animation(self):
        self.animation_active = True
        return True

    def stop_animation(self):
        self.animation_active = False
        return True

    def update_qr_code(self, qr_data):
        return {"status": "mock_updated", "qr_data": qr_data}
