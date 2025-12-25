"""Locust load testing for voice service.

Run with:
    locust -f tests/locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, between, task


class VoiceNoteUser(HttpUser):
    """Simulates voice note generation requests."""

    wait_time = between(1, 3)

    @task(10)
    def generate_short_audio(self):
        """Generate short audio (voice note)."""
        self.client.post(
            "/v1/audio/speech",
            json={
                "input": "Hello! This is a short voice note.",
                "voice": "default",
                "response_format": "pcm",
            },
            name="/v1/audio/speech [short]",
        )

    @task(5)
    def generate_medium_audio(self):
        """Generate medium length audio."""
        self.client.post(
            "/v1/audio/speech",
            json={
                "input": "This is a longer voice message that contains more content. "
                "It should take a bit more time to generate but still be reasonable.",
                "voice": "default",
                "response_format": "pcm",
            },
            name="/v1/audio/speech [medium]",
        )

    @task(2)
    def generate_streaming(self):
        """Generate streaming audio."""
        self.client.post(
            "/v1/audio/speech",
            json={
                "input": "Streaming audio test message.",
                "voice": "default",
                "response_format": "pcm",
                "stream": True,
            },
            name="/v1/audio/speech [stream]",
        )

    @task(3)
    def health_check(self):
        """Check service health."""
        self.client.get("/health")

    @task(1)
    def list_voices(self):
        """List available voices."""
        self.client.get("/v1/voices")


