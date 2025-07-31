"""
QR code animation functionality for lukhas-id mobile platform.

ΛTAG: mobile_platform, qr_code, animation, lukhas_id
"""

import time
import base64
from io import BytesIO
from typing import Dict, Any, Optional
import uuid

import qrcode


class QRCodeAnimator:
    """QR code animation manager for mobile platform."""

    def __init__(self, refresh_interval: float = 1.0):
        self.refresh_interval = refresh_interval
        self.animation_active = False
        self.current_qr_data = None
        self.animation_frame = 0
        self.last_update = None

    def generate_glyph(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Generate unique QR glyph encoded as base64 string."""
        data = f"{user_id}:{session_id or uuid.uuid4().hex}:{time.time()}"
        qr = qrcode.QRCode(border=1)
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def start_animation(self):
        """Start QR code animation."""
        self.animation_active = True
        self.last_update = time.time()
        return True

    def stop_animation(self):
        """Stop QR code animation."""
        self.animation_active = False
        self.animation_frame = 0
        return True

    def update_qr_code(self, qr_data: Optional[str] = None, user_id: str = "anon") -> Dict[str, Any]:
        """Update QR code data and refresh animation."""
        if qr_data is None:
            qr_data = self.generate_glyph(user_id)
        self.current_qr_data = qr_data
        self.animation_frame += 1
        self.last_update = time.time()

        return {
            "status": "updated",
            "qr_data": qr_data,
            "frame": self.animation_frame,
            "timestamp": self.last_update,
        }

    def get_current_frame(self) -> Dict[str, Any]:
        """Get current animation frame data."""
        return {
            "frame": self.animation_frame,
            "qr_data": self.current_qr_data,
            "active": self.animation_active,
            "last_update": self.last_update,
        }

    def is_animation_active(self) -> bool:
        """Check if animation is currently running."""
        return self.animation_active

    def should_refresh(self) -> bool:
        """Check if QR code should be refreshed based on interval."""
        if not self.last_update:
            return True

        return (time.time() - self.last_update) >= self.refresh_interval

    def get_animation_status(self) -> Dict[str, Any]:
        """Get comprehensive animation status."""
        return {
            "active": self.animation_active,
            "frame": self.animation_frame,
            "qr_data": self.current_qr_data,
            "last_update": self.last_update,
            "refresh_interval": self.refresh_interval,
            "should_refresh": self.should_refresh(),
        }
