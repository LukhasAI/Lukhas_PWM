"""
Mobile WebSocket client for lukhas-id mobile platform.

ΛTAG: mobile_platform, websocket, lukhas_id
"""


class MobileWebSocketClient:
    """WebSocket client for mobile platform communication."""

    def __init__(self, url=None):
        self.url = url or "ws://localhost:8080"
        self.connected = False
        self.messages = []

    def connect(self):
        """Connect to the WebSocket server."""
        self.connected = True
        return True

    def disconnect(self):
        """Disconnect from the WebSocket server."""
        self.connected = False
        return True

    def send_message(self, message):
        """Send a message through the WebSocket connection."""
        if not self.connected:
            return {"status": "error", "message": "Not connected"}

        self.messages.append(message)
        return {"status": "sent", "message": message}

    def receive_message(self):
        """Receive a message from the WebSocket connection."""
        if not self.connected:
            return {"status": "error", "message": "Not connected"}

        if self.messages:
            return {"status": "received", "message": self.messages.pop(0)}
        return {"status": "no_messages"}

    def is_connected(self):
        """Check if the WebSocket connection is active."""
        return self.connected
