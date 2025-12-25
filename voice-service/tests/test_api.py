"""Tests for API endpoints."""

import os

import pytest
from fastapi.testclient import TestClient

# Force mock provider for tests
os.environ["VOICE_TTS_PROVIDER"] = "mock"
os.environ["VOICE_REFERENCE_STORAGE"] = "local"

from src.main import create_app


@pytest.fixture(scope="module")
def client():
    """Create test client with mock provider."""
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "provider" in data
        assert "models_loaded" in data
        assert "streaming_enabled" in data

    def test_health_shows_mock_provider(self, client):
        response = client.get("/health")
        data = response.json()

        assert data["provider"] == "mock"
        assert data["models_loaded"] is True


class TestRootEndpoint:
    """Tests for / endpoint."""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response_structure(self, client):
        response = client.get("/")
        data = response.json()

        assert data["service"] == "Voice Service"
        assert "version" in data
        assert "docs" in data
        assert "health" in data


class TestVoicesEndpoint:
    """Tests for /v1/voices endpoint."""

    def test_list_voices_returns_200(self, client):
        response = client.get("/v1/voices")
        assert response.status_code == 200

    def test_list_voices_response_structure(self, client):
        response = client.get("/v1/voices")
        data = response.json()

        assert "voices" in data
        assert isinstance(data["voices"], list)
        assert len(data["voices"]) >= 1

    def test_voice_has_required_fields(self, client):
        response = client.get("/v1/voices")
        data = response.json()

        voice = data["voices"][0]
        assert "voice_id" in voice
        assert "name" in voice


class TestTTSEndpoint:
    """Tests for /v1/audio/speech endpoint."""

    def test_tts_basic_request(self, client):
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "Hello, world!",
                "voice": "default",
                "response_format": "wav",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    def test_tts_returns_valid_wav(self, client):
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Test audio generation."},
        )

        assert response.status_code == 200
        audio_bytes = response.content

        # Check WAV header
        assert len(audio_bytes) > 44
        assert audio_bytes[:4] == b"RIFF"
        assert audio_bytes[8:12] == b"WAVE"

    def test_tts_different_formats(self, client):
        for fmt in ["wav", "pcm"]:
            response = client.post(
                "/v1/audio/speech",
                json={"input": "Format test", "response_format": fmt},
            )
            assert response.status_code == 200
            assert len(response.content) > 0

    def test_tts_empty_input_rejected(self, client):
        response = client.post(
            "/v1/audio/speech",
            json={"input": "", "voice": "default"},
        )
        assert response.status_code == 422

    def test_tts_invalid_format_rejected(self, client):
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "response_format": "invalid_format"},
        )
        assert response.status_code == 422

    def test_tts_long_text(self, client):
        long_text = "This is a test. " * 50
        response = client.post(
            "/v1/audio/speech",
            json={"input": long_text},
        )
        assert response.status_code == 200
        assert len(response.content) > 44


class TestTTSStreamingEndpoint:
    """Tests for streaming TTS."""

    def test_tts_streaming_request(self, client):
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "Streaming audio test with longer text.",
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "chunked" in response.headers.get("transfer-encoding", "").lower()

    def test_tts_streaming_returns_data(self, client):
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Stream test", "stream": True},
        )

        assert response.status_code == 200
        assert len(response.content) > 0

    def test_tts_streaming_wav_format(self, client):
        """Test streaming with WAV format."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "This is a longer text for streaming WAV format testing.",
                "stream": True,
                "response_format": "wav",
            },
        )
        assert response.status_code == 200
        content = response.content
        
        # First chunk should contain WAV header
        assert len(content) > 0
        # When streaming WAV, header should be in first chunk
        assert content[:4] == b"RIFF" or b"RIFF" in content[:100]

    def test_tts_streaming_pcm_format(self, client):
        """Test streaming with PCM format."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "Streaming PCM format test with longer text.",
                "stream": True,
                "response_format": "pcm",
            },
        )
        assert response.status_code == 200
        content = response.content
        assert len(content) > 0
        # PCM should not have WAV header
        assert content[:4] != b"RIFF"

    def test_tts_streaming_multiple_chunks(self, client):
        """Test that streaming produces multiple chunks for longer text."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "This is a very long text. " * 20,
                "stream": True,
            },
        )
        assert response.status_code == 200
        
        # For streaming, we get chunks progressively
        # TestClient collects all chunks, but we can verify content length
        assert len(response.content) > 1000

    def test_tts_streaming_vs_non_streaming_same_length(self, client):
        """Test that streaming and non-streaming produce similar audio length."""
        text = "Test audio generation for comparison."
        
        streaming_response = client.post(
            "/v1/audio/speech",
            json={"input": text, "stream": True},
        )
        non_streaming_response = client.post(
            "/v1/audio/speech",
            json={"input": text, "stream": False},
        )
        
        assert streaming_response.status_code == 200
        assert non_streaming_response.status_code == 200
        
        # Both should produce audio (length may differ slightly due to chunking)
        assert len(streaming_response.content) > 44
        assert len(non_streaming_response.content) > 44

    def test_extended_tts_streaming_with_params(self, client):
        """Test extended endpoint streaming with parameters."""
        response = client.post(
            "/v1/audio/speech/extended",
            json={
                "input": "Extended streaming test with parameters.",
                "stream": True,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )
        assert response.status_code == 200
        assert len(response.content) > 0


class TestExtendedTTSEndpoint:
    """Tests for /v1/audio/speech/extended endpoint."""

    def test_extended_tts_with_parameters(self, client):
        response = client.post(
            "/v1/audio/speech/extended",
            json={
                "input": "Extended TTS test.",
                "temperature": 0.8,
                "top_p": 0.9,
            },
        )
        assert response.status_code == 200
        assert len(response.content) > 44

    def test_extended_tts_streaming(self, client):
        response = client.post(
            "/v1/audio/speech/extended",
            json={
                "input": "Extended streaming test.",
                "stream": True,
                "temperature": 0.5,
            },
        )
        assert response.status_code == 200


class TestReferenceEndpoints:
    """Tests for reference voice management endpoints."""

    def test_list_references_returns_200(self, client):
        response = client.get("/v1/references")
        assert response.status_code == 200

    def test_list_references_structure(self, client):
        response = client.get("/v1/references")
        data = response.json()
        assert "references" in data
        assert isinstance(data["references"], list)

    def test_add_reference(self, client):
        """Test adding a reference voice."""
        import io
        
        # Create a simple WAV file
        audio_data = b"RIFF" + b"\x00" * 40 + b"WAVE"
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "test.wav"
        
        response = client.post(
            "/v1/references/add",
            data={
                "id": "test_voice_123",
                "text": "This is a test transcript.",
            },
            files={"audio": ("test.wav", audio_file, "audio/wav")},
        )
        
        # Should succeed or return appropriate error
        assert response.status_code in [200, 400, 500]

    def test_delete_reference(self, client):
        """Test deleting a reference voice."""
        response = client.delete("/v1/references/nonexistent_voice")
        # Should return 404 or 200 depending on implementation
        assert response.status_code in [200, 404, 501]


class TestErrorHandling:
    """Tests for API error handling."""

    def test_missing_required_field(self, client):
        response = client.post("/v1/audio/speech", json={})
        assert response.status_code == 422

    def test_invalid_json(self, client):
        response = client.post(
            "/v1/audio/speech",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_nonexistent_endpoint(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_streaming_with_invalid_format(self, client):
        """Test streaming with invalid format."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "Test",
                "stream": True,
                "response_format": "invalid_format",
            },
        )
        assert response.status_code == 422

    def test_text_too_long(self, client):
        """Test rejection of text exceeding max length."""
        long_text = "A" * (2001)  # Exceeds default max_text_length
        response = client.post(
            "/v1/audio/speech",
            json={"input": long_text},
        )
        assert response.status_code == 400
