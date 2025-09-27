import asyncio
import base64
import json
import os
import tempfile
import time
from datetime import datetime
from typing import Dict, Optional
from urllib.parse import urljoin

import numpy as np
import soundfile as sf
import transformers
from fastapi import FastAPI, WebSocket, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline
from pydantic import BaseModel
import torch

class VoiceAssistant:
    VOICES = {
        "Bella (US Female)": {"code": "af_bella", "lang_code": "a"},
        "Nicole (US Female)": {"code": "af_nicole", "lang_code": "a"},
        "Michael (US Male)": {"code": "am_michael", "lang_code": "a"},
        "Emma (UK Female)": {"code": "bf_emma", "lang_code": "b"},
        "George (UK Male)": {"code": "bm_george", "lang_code": "b"}
    }

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        print("=== Initializing VoiceAssistant ===")

        print("Loading Ultravox model...")
        try:
            # Clear GPU cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

            # Use smaller v0_4 model to avoid OOM
            self.pipe = transformers.pipeline(
                model='fixie-ai/ultravox-v0_6-llama-3_1-8b',
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use half precision to save memory
                device_map="auto"
            )
            print("✓ Ultravox model loaded successfully")

            if torch.cuda.is_available():
                print(f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        except Exception as e:
            print(f"✗ Failed to load Ultravox model: {e}")
            raise

        print("Loading UltraVAD model...")
        try:
            self.vad_pipe = transformers.pipeline(
                model='fixie-ai/ultraVAD',
                trust_remote_code=True
            )
            print("✓ UltraVAD model loaded successfully")
        except Exception as e:
            print(f"✗ Could not load UltraVAD model: {e}")
            print("  UltraVAD functionality will use fallback logic")
            self.vad_pipe = None

        print("Loading Turn-taking model...")
        try:
            # Try to load as a standard transformers model first
            from transformers import AutoModel, AutoTokenizer
            self.turntaking_tokenizer = AutoTokenizer.from_pretrained(
                'fixie-ai/turntaking-pretraining-it-multilingual-3c',
                trust_remote_code=True
            )
            self.turntaking_model = AutoModel.from_pretrained(
                'fixie-ai/turntaking-pretraining-it-multilingual-3c',
                trust_remote_code=True
            )
            print("✓ Turn-taking model loaded successfully as AutoModel")
        except Exception as e:
            print(f"✗ Could not load turn-taking model as AutoModel: {e}")
            print("  Turn-taking functionality will use fallback logic")
            self.turntaking_tokenizer = None
            self.turntaking_model = None

        self.tts_pipelines: Dict[str, KPipeline] = {}
        self.current_voice = list(self.VOICES.keys())[0]
        self.system_prompt = system_prompt
        self.state = "idle"

        # Initialize basic VAD as fallback
        print("Loading fallback Silero VAD...")
        self.fallback_vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.fallback_vad_model.eval()

        # VAD parameters
        self.sample_rate = 16000
        self.vad_window_size = 512  # 32ms window for 16kHz
        self.speech_threshold = 0.2   # Very sensitive for general speech detection
        self.eot_threshold = 0.1      # UltraVAD threshold (start with recommended 0.1)
        self.min_speech_duration_ms = 200  # Reduced minimum speech duration
        self.min_silence_duration_ms = 300  # Minimum silence duration before timeout

        # Conversation context for turn-taking
        self.conversation_turns = []
        self.max_context_turns = 10  # Keep last 10 turns for context

        # Agent state tracking
        self.agent_is_speaking = False
        self.interruption_detected = False
        self.interruption_threshold = 0.1  # Extremely low threshold for interruption detection

        # Audio buffers
        self.audio_window = []
        self.speech_probs = []
        self.current_speech = []
        self.speech_timestamps = []

        # Buffers
        self.vad_buffer = []
        self.speech_buffer = []
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_counter = 0

        print("=== VoiceAssistant initialization complete ===")
        print(f"Models loaded:")
        print(f"  - Ultravox: ✓")
        print(f"  - UltraVAD: {'✓' if self.vad_pipe else '✗ (using fallback)'}")
        print(f"  - Turn-taking: {'✓' if self.turntaking_model else '✗ (using fallback)'}")
        print(f"  - Fallback VAD: ✓")

    def get_tts_pipeline(self, voice_name: str) -> KPipeline:
        if voice_name not in self.tts_pipelines:
            voice_config = self.VOICES[voice_name]
            self.tts_pipelines[voice_name] = KPipeline(lang_code=voice_config["lang_code"])
        return self.tts_pipelines[voice_name]

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Basic VAD detection using fallback Silero VAD"""
        try:
            # Add chunk to audio window
            self.audio_window.extend(audio_chunk.tolist())

            # Process complete windows of exactly 512 samples
            speech_probs = []
            while len(self.audio_window) >= self.vad_window_size:
                # Get window of exactly 512 samples
                window = np.array(self.audio_window[:self.vad_window_size])
                self.audio_window = self.audio_window[self.vad_window_size:]

                # Ensure we have exactly 512 samples before processing
                if len(window) == self.vad_window_size:
                    # Convert to tensor and reshape for Silero VAD (batch_size=1, num_samples=512)
                    tensor = torch.FloatTensor(window).unsqueeze(0)

                    # Get speech probability
                    speech_prob = self.fallback_vad_model(tensor, self.sample_rate).item()
                    speech_probs.append(speech_prob)

            # Update running probabilities list
            if speech_probs:
                self.speech_probs.extend(speech_probs)

                # Keep only recent probabilities (last 1 second)
                window_size = int(self.sample_rate / self.vad_window_size)
                if len(self.speech_probs) > window_size:
                    self.speech_probs.pop(0)

            # Calculate moving average of speech probabilities
            if not self.speech_probs:
                return False

            avg_speech_prob = sum(self.speech_probs) / len(self.speech_probs)
            has_speech = avg_speech_prob > self.speech_threshold

            # Minimal logging - only when speech is detected
            if has_speech and len(self.speech_probs) % 10 == 0 and not self.agent_is_speaking:
                print(f"🎵 Speech detected: {avg_speech_prob:.3f}")

            return has_speech

        except Exception as e:
            print(f"Basic VAD error: {e}")
            return False

    def check_end_of_turn(self, audio: np.ndarray) -> bool:
        """Check if user finished speaking using UltraVAD context-aware endpointing"""
        # If UltraVAD is not available, use fallback logic
        if self.vad_pipe is None:
            print("UltraVAD not available, using silence-based detection")
            return self.silence_counter >= self.min_silence_duration_ms

        try:
            # Ensure audio is the right length (UltraVAD may expect specific lengths)
            if len(audio) < 1600:  # Less than 100ms of audio
                print(f"Audio too short for UltraVAD ({len(audio)} samples), using fallback")
                return self.silence_counter >= self.min_silence_duration_ms

            # Prepare conversation context - include system prompt for better context
            recent_turns = [{"role": "system", "content": self.system_prompt}]
            if self.conversation_turns:
                recent_turns.extend(self.conversation_turns[-self.max_context_turns:])

            # Use UltraVAD for end-of-turn detection
            inputs = {
                "audio": audio,
                "turns": recent_turns,
                "sampling_rate": self.sample_rate
            }

            # UltraVAD processing...

            # Get end-of-turn probability
            output = self.vad_pipe(inputs)

            # Handle different possible output formats
            if hasattr(output, 'end_of_turn_probability'):
                eot_probability = output.end_of_turn_probability
            elif isinstance(output, dict):
                eot_probability = output.get('end_of_turn_probability',
                                          output.get('probability',
                                          output.get('score', 0)))
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                eot_probability = output[0] if isinstance(output[0], (int, float)) else 0
            else:
                print(f"Unexpected UltraVAD output format: {type(output)}")
                eot_probability = 0

            # Only log if significant probability
            if eot_probability > 0.05:
                print(f"End-of-turn: {eot_probability:.3f}")
            return eot_probability > self.eot_threshold

        except Exception as e:
            print(f"UltraVAD error, using fallback: {e}")
            # Fallback to simple silence-based detection
            return self.silence_counter >= self.min_silence_duration_ms

    def should_respond_with_turntaking(self, audio: np.ndarray, context: list) -> bool:
        """Use turn-taking model to decide if agent should respond"""
        # If turn-taking model is not available, use fallback logic
        if self.turntaking_model is None or self.turntaking_tokenizer is None:
            print("Turn-taking model not available, using fallback logic")
            # Simple fallback: respond if we have enough speech duration
            audio_duration_ms = len(audio) * 1000 / self.sample_rate
            should_respond = audio_duration_ms >= self.min_speech_duration_ms
            print(f"Fallback turn-taking decision: {'respond' if should_respond else 'wait'}")
            return should_respond

        try:
            # For now, use simple logic since the model interface is unclear
            # In a production system, you'd implement the proper model inference here
            print("Using simplified turn-taking logic")

            # Consider conversation length - respond more readily in early conversation
            context_length = len(context)
            audio_duration_ms = len(audio) * 1000 / self.sample_rate

            # More responsive for shorter conversations, more careful for longer ones
            threshold = self.min_speech_duration_ms + (context_length * 50)  # Increase threshold over time
            should_respond = audio_duration_ms >= threshold

            print(f"Turn-taking decision (context_len={context_length}): {'respond' if should_respond else 'wait'}")
            return should_respond

        except Exception as e:
            print(f"Turn-taking model error, defaulting to respond: {e}")
            return True

    def check_interruption(self, audio_chunk: np.ndarray) -> bool:
        """Check if user is interrupting agent speech - very sensitive detection"""
        if not self.agent_is_speaking:
            return False

        try:
            # Direct VAD check on current chunk - much more sensitive
            if len(audio_chunk) >= 512:  # We need at least 512 samples
                # Take the last 512 samples for VAD
                vad_chunk = audio_chunk[-512:]
                tensor = torch.FloatTensor(vad_chunk).unsqueeze(0)
                speech_prob = self.fallback_vad_model(tensor, self.sample_rate).item()

                # Only log when close to or above threshold
                if speech_prob > 0.05:  # Only log when there's significant audio
                    print(f"🎤 Interruption check: {speech_prob:.3f}")

                if speech_prob > self.interruption_threshold:
                    print(f"🚨 INTERRUPTION DETECTED! Speech probability: {speech_prob:.3f}")
                    self.interruption_detected = True
                    return True

        except Exception as e:
            print(f"Interruption detection error: {e}")

        return False

    def add_audio(self, audio_chunk: np.ndarray) -> bool:
        """Enhanced audio processing with UltraVAD context-aware endpointing"""
        # Interruption handling is now done at WebSocket level
        has_speech = self.is_speech(audio_chunk)

        # Calculate durations
        chunk_duration_ms = len(audio_chunk) * 1000 / self.sample_rate

        if has_speech:
            if not self.current_speech:  # Start of speech
                print("Speech started...")
            self.current_speech.extend(audio_chunk.tolist())
            self.silence_counter = 0
        else:
            if self.current_speech:  # Potential end of speech
                self.silence_counter += chunk_duration_ms
                # Don't add silence to speech buffer - keep it clean
                # self.current_speech.extend(audio_chunk.tolist())

        # Only check for end-of-turn if we have substantial speech and some silence
        should_check_eot = (
            len(self.current_speech) > 0 and
            len(self.current_speech) * 1000 / self.sample_rate >= self.min_speech_duration_ms and
            self.silence_counter >= 100  # At least 100ms of silence before checking
        )

        if should_check_eot:
            current_audio = np.array(self.current_speech)

            # Only call UltraVAD occasionally, not on every chunk
            if self.silence_counter % 200 < chunk_duration_ms:  # Check every ~200ms of silence
                if self.check_end_of_turn(current_audio):
                    # Also check turn-taking model for decision
                    if self.should_respond_with_turntaking(current_audio, self.conversation_turns):
                        print(f"End of turn detected - duration: {len(self.current_speech) * 1000 / self.sample_rate:.0f}ms")
                        self.audio_buffer = self.current_speech.copy()
                        self.current_speech = []
                        self.silence_counter = 0
                        self.speech_probs = []
                        return True
                    else:
                        print("Turn-taking model suggests waiting...")

        # Reset if silence is too long (fallback)
        if self.silence_counter > self.min_silence_duration_ms * 2:  # Reduced from 3x
            if self.current_speech:
                print("Timeout reached, processing speech...")
                self.audio_buffer = self.current_speech.copy()
                self.current_speech = []
                self.silence_counter = 0
                self.speech_probs = []
                return True
            self.current_speech = []
            self.silence_counter = 0
            self.speech_probs = []

        return False

    def get_audio(self) -> np.ndarray:
        """Get accumulated audio and clear buffer"""
        audio = np.array(self.audio_buffer)
        self.audio_buffer = []
        return audio

    def add_conversation_turn(self, role: str, content: str):
        """Add a conversation turn for context tracking"""
        turn = {"role": role, "content": content}
        self.conversation_turns.append(turn)

        # Keep only recent turns
        if len(self.conversation_turns) > self.max_context_turns:
            self.conversation_turns = self.conversation_turns[-self.max_context_turns:]

        # Removed noisy context logging

    def get_conversation_context(self) -> list:
        """Get conversation context for models"""
        context = [{"role": "system", "content": self.system_prompt}]
        context.extend(self.conversation_turns)
        return context

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """Simple transcription to debug what user actually said"""
        try:
            # Use Ultravox just for transcription (no response generation)
            result = self.pipe({
                'audio': audio,
                'turns': [{"role": "system", "content": "Transcribe what the user said. Only return the transcription, nothing else."}],
                'sampling_rate': self.sample_rate
            }, max_new_tokens=30)

            transcription = result[0] if isinstance(result, list) else str(result)
            return transcription.strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return "[transcription failed]"

class CallConfig(BaseModel):
    systemPrompt: str
    temperature: float = 0.8
    voice: Optional[str] = None
    medium: dict = {
        "serverWebSocket": {
            "inputSampleRate": 16000,
            "outputSampleRate": 16000,
            "clientBufferSizeMs": 30000
        }
    }
    selectedTools: list = []
    firstSpeaker: str = "FIRST_SPEAKER_AGENT"
    initialOutputMedium: str = "MESSAGE_MEDIUM_SPEECH"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global assistant instance
assistant = None

def get_base_url(request: Request) -> str:
    """Get base URL from request"""
    host = request.headers.get("host", "localhost:7860")
    scheme = request.headers.get("x-forwarded-proto", "http")
    return f"{scheme}://{host}"

@app.post("/api/calls")
async def create_call(
    config: CallConfig,
    request: Request,
    x_api_key: Optional[str] = Header(None)
):
    # Optional: Validate API key if needed
    if os.getenv("REQUIRE_API_KEY"):
        if not x_api_key or x_api_key != os.getenv("ULTRAVOX_API_KEY"):
            return {"error": "Invalid API key"}, 401

    global assistant
    try:
        assistant = VoiceAssistant(system_prompt=config.systemPrompt)
        if config.voice:
            assistant.current_voice = config.voice
    except Exception as e:
        print(f"Error initializing VoiceAssistant: {e}")
        return {"error": f"Failed to initialize voice assistant: {str(e)}"}, 500
    
    # Generate a unique call ID
    call_id = f"call_{int(time.time())}"
    
    # Construct join URL using the request's base URL
    base_url = get_base_url(request)
    ws_scheme = "wss" if base_url.startswith("https") else "ws"
    join_url = f"{ws_scheme}://{request.headers['host']}/api/calls/{call_id}/join"
    
    return {
        "callId": call_id,
        "joinUrl": join_url,
        "status": "success"
    }

@app.websocket("/api/calls/{call_id}/join")
async def join_call(websocket: WebSocket, call_id: str):
    if not assistant:
        await websocket.close(code=4000, reason="No active call")
        return

    await websocket.accept()
    print(f"WebSocket connection established for call {call_id}")
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "state",
            "state": "speaking",
            "timestamp": int(time.time() * 1000)
        })

        # Send initial greeting - keep it brief
        initial_text = "Hi! How can I help?"
        await websocket.send_json({
            "type": "transcript",
            "role": "agent",
            "text": initial_text,
            "final": True,
            "timestamp": int(time.time() * 1000)
        })

        # Generate and send initial TTS audio
        # Initial greeting state
        assistant.agent_is_speaking = True
        tts_pipeline = assistant.get_tts_pipeline(assistant.current_voice)
        audio_segments = []
        for _, _, audio_data in tts_pipeline(initial_text,
                                           voice=assistant.VOICES[assistant.current_voice]["code"],
                                           speed=1):
            audio_segments.append(audio_data)

        if audio_segments:
            combined_audio = np.concatenate(audio_segments)
            audio_int16 = (combined_audio * 32768).astype(np.int16)
            await websocket.send_bytes(audio_int16.tobytes())

        # Agent finished initial greeting
        # Initial greeting finished
        assistant.agent_is_speaking = False

        # Switch to listening state
        await websocket.send_json({
            "type": "state",
            "state": "listening",
            "timestamp": int(time.time() * 1000)
        })

        # Adjust VAD parameters
        assistant.min_speech_frames = 5
        assistant.silence_frames = 10

        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                
                if "bytes" in message:
                    audio_data = message["bytes"]
                    audio_chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                    # Removed excessive logging - only log important events

                    # ALWAYS check for interruption first, regardless of other processing
                    if assistant.check_interruption(audio_chunk):
                        print("🛑 INTERRUPTION! Stopping agent speech immediately")
                        assistant.agent_is_speaking = False
                        assistant.interruption_detected = False

                        # Clear any current speech processing
                        assistant.current_speech = []
                        assistant.silence_counter = 0
                        assistant.speech_probs = []

                        await websocket.send_json({
                            "type": "state",
                            "state": "listening",
                            "timestamp": int(time.time() * 1000)
                        })
                        await websocket.send_json({
                            "type": "transcript",
                            "role": "system",
                            "text": "[Agent interrupted]",
                            "final": True,
                            "timestamp": int(time.time() * 1000)
                        })
                        continue

                    # Only process speech if agent is not speaking
                    if assistant.agent_is_speaking:
                        continue  # Skip processing while agent speaks

                    if assistant.add_audio(audio_chunk):
                        print("\n🎤 USER SPOKE - Processing speech...")
                        audio = assistant.get_audio()

                        if len(audio) < 1600:  # Skip if too short
                            print("❌ Audio too short, skipping")
                            continue

                        # First, let's see what the user actually said
                        user_transcription = assistant.transcribe_audio(audio)
                        print(f"📝 USER SAID: '{user_transcription}'")

                        await websocket.send_json({
                            "type": "state",
                            "state": "thinking",
                            "timestamp": int(time.time() * 1000)
                        })

                        # Show user transcription to client
                        await websocket.send_json({
                            "type": "transcript",
                            "role": "user",
                            "text": user_transcription,
                            "final": True,
                            "timestamp": int(time.time() * 1000)
                        })

                        # Use conversation context for Ultravox with shorter responses
                        context = assistant.get_conversation_context()
                        # Add instruction for brief responses
                        context.append({"role": "system", "content": "Keep responses brief and conversational. 1-2 sentences max."})

                        result = assistant.pipe({
                            'audio': audio,
                            'turns': context,
                            'sampling_rate': 16000
                        }, max_new_tokens=50)  # Reduced from 200 for brevity

                        text_response = result[0] if isinstance(result, list) else str(result)
                        print(f"🤖 AGENT RESPONSE: '{text_response}'")

                        # Add user and agent turns to conversation context
                        assistant.add_conversation_turn("user", user_transcription)
                        assistant.add_conversation_turn("assistant", text_response)

                        await websocket.send_json({
                            "type": "transcript",
                            "role": "agent",
                            "text": text_response,
                            "final": True,
                            "timestamp": int(time.time() * 1000)
                        })
                        
                        await websocket.send_json({
                            "type": "state",
                            "state": "speaking",
                            "timestamp": int(time.time() * 1000)
                        })

                        # Set agent speaking state for interruption detection
                        # Agent speaking state set for interruption
                        assistant.agent_is_speaking = True
                        assistant.interruption_detected = False

                        audio_segments = []
                        for _, _, audio_data in tts_pipeline(text_response,
                                                           voice=assistant.VOICES[assistant.current_voice]["code"],
                                                           speed=1):
                            audio_segments.append(audio_data)

                        if audio_segments:
                            combined_audio = np.concatenate(audio_segments)
                            audio_int16 = (combined_audio * 32768).astype(np.int16)
                            await websocket.send_bytes(audio_int16.tobytes())

                        # Agent finished speaking
                        assistant.agent_is_speaking = False

                        await websocket.send_json({
                            "type": "state",
                            "state": "listening",
                            "timestamp": int(time.time() * 1000)
                        })
                elif "type" in message and message["type"] == "close":
                    print(f"Client requested close for call {call_id}")
                    break

            except asyncio.TimeoutError:
                # Check if client is still connected
                try:
                    await websocket.send_json({"type": "ping"})
                except:
                    print(f"Client disconnected (timeout) for call {call_id}")
                    break
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                print(f"Current buffer sizes - VAD: {len(assistant.vad_buffer)}, Speech: {len(assistant.speech_buffer)}")  # Debug log
                if "disconnect" in str(e).lower() or "closed" in str(e).lower():
                    print(f"Client disconnected for call {call_id}")
                    break
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                        "timestamp": int(time.time() * 1000)
                    })
                except:
                    break

    except Exception as e:
        print(f"WebSocket connection error for call {call_id}: {str(e)}")
    finally:
        print(f"Cleaning up call {call_id}")
        try:
            await websocket.close()
        except:
            pass
        # Clear assistant's state
        assistant.audio_buffer = []
        assistant.is_speaking = False
        assistant.speech_buffer = []

def main():
    print("Starting Speech Server...")
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=7860,
        ssl_keyfile=os.getenv("SSL_KEYFILE"),
        ssl_certfile=os.getenv("SSL_CERTFILE")
    )

if __name__ == "__main__":
    main() 