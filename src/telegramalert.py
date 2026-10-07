import os
import time
import requests
import cv2
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger("TelegramNotifier")

class TelegramNotifier:
    def __init__(self, min_strikes=3, cooldown_seconds=60):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.default_chat_id = os.getenv("TELEGRAM_GROUP_ID")
        
        # Strike configuration
        self.min_strikes = min_strikes
        self.cooldown_seconds = cooldown_seconds
        
        # Tracking states
        # Format: { "incident_key": {"count": int, "last_sent": timestamp, "last_seen": timestamp} }
        self.alert_state = {}

    def _generate_incident_key(self, alert):
        """Generates a unique key for the alert to track its frequency."""
        alert_type = alert.get("type", "UNKNOWN")
        track_id = alert.get("track_id", None)
        
        if track_id is not None:
            return f"{alert_type}_track_{track_id}"
        return f"{alert_type}_general"

    def _cleanup_stale_alerts(self, now):
        """Removes alerts that haven't been seen recently to free memory."""
        keys_to_delete = []
        for key, state in self.alert_state.items():
            if now - state["last_seen"] > self.cooldown_seconds * 2:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.alert_state[key]

    def process_alert(self, alert, frame_rgb, target_chat_id=None):
        """
        Process an incoming alert. If the threshold is met, send it to Telegram.
        frame_rgb should be the image array (RGB or BGR is fine, we encode it directly).
        """
        if not self.bot_token:
            logger.error("TELEGRAM_BOT_TOKEN not found in environment.")
            return

        chat_id = target_chat_id or self.default_chat_id
        if not chat_id:
            logger.error("No target chat_id provided (set TELEGRAM_GROUP_ID in .env).")
            return

        now = time.time()
        self._cleanup_stale_alerts(now)

        key = self._generate_incident_key(alert)
        state = self.alert_state.get(key, {"count": 0, "last_sent": 0.0, "last_seen": now})
        
        state["count"] += 1
        state["last_seen"] = now
        
        # Check if we should send
        alert_conf = alert.get("conf", 0.0)
        is_high_conf = alert_conf >= 0.70
        
        if is_high_conf or state["count"] >= self.min_strikes:
            if now - state["last_sent"] >= self.cooldown_seconds:
                reason = "high confidence > 70%" if is_high_conf else f"strike count reached ({state['count']})"
                logger.info(f"Threshold met for {key} via {reason}. Sending alert to Telegram...")
                self._send_to_telegram(chat_id, alert, frame_rgb)
                
                # Update last sent time and reset count
                state["last_sent"] = now
                state["count"] = 0 # reset count after sending
        
        self.alert_state[key] = state

    def _send_to_telegram(self, chat_id, alert, frame):
        import threading
        
        def upload_task():
            try:
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logger.error("Failed to encode frame to JPEG for Telegram.")
                    return

                # Prepare Telegram API request
                url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
                
                message = alert.get("message", "Anomaly Detected!")
                caption = f"🚨 *SECURITY ALERT* 🚨\n\n*Type:* {alert.get('type')}\n*Message:* {message}\n*Time:* {time.strftime('%Y-%m-%d %H:%M:%S')}"

                # Files payload
                files = {
                    'photo': ('alert.jpg', buffer.tobytes(), 'image/jpeg')
                }
                # Data payload
                data = {
                    'chat_id': chat_id,
                    'caption': caption,
                    'parse_mode': 'Markdown'
                }

                response = requests.post(url, data=data, files=files, timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"Successfully sent alert screenshot to Telegram chat {chat_id}")
                else:
                    logger.error(f"Telegram API Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"Exception while sending to Telegram: {e}")

        # Start the upload in a background thread so it doesn't block AI/video processing
        threading.Thread(target=upload_task, daemon=True).start()
