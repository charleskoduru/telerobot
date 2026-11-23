import asyncio
import json
import ssl
import threading
from pathlib import Path
from queue import Empty, Full, Queue
import websockets

from telerobot import PACKAGE_DIR


def create_ssl_context(cert_file: str, key_file: str):
    """Create SSL context for WSS (secure WebSocket)."""
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(cert_file, key_file)
    return ssl_context


class WebSocketServer:
    name = "websocket_server"

    def __init__(self, use_ssl: bool = True, cert_file: str = None, key_file: str = None):
        self.connected = False
        self.last_observation = None
        # Separate one-shot commands from continuously overwritten VR frames.
        self._alignment_requests = Queue(maxsize=1)
        self._server = None
        self._loop = None
        self._thread = None
        self._ready_event = threading.Event()  # to signal readiness
        self._clients = set()
        self._runtime_status = {}
        self._startup_error = None
        self._stop_requested = threading.Event()
        
        # SSL configuration
        self.use_ssl = use_ssl
        self.ssl_context = None
        
        # Default certificate paths (relative to CWD)
        if cert_file is None:
            cert_file = Path("ssl_cert") / "server.crt"
        if key_file is None:
            key_file = Path("ssl_cert") / "server.key"
        
        if use_ssl and Path(cert_file).exists() and Path(key_file).exists():
            self.ssl_context = create_ssl_context(str(cert_file), str(key_file))
            print(f"✅ SSL enabled for WebSocket server")
        elif use_ssl:
            print(f"⚠️ SSL certificates not found at {cert_file}, {key_file}")
            print("   WebSocket server will run without SSL (ws://)")
            self.use_ssl = False

    @property
    def is_connected(self) -> bool:
        return self.connected

    async def on_observation_received(self, websocket):
        self._clients.add(websocket)
        try:
            if self._runtime_status:
                await websocket.send(json.dumps({
                    "type": "runtime_status",
                    **self._runtime_status,
                }))
            async for message in websocket:
                payload = json.loads(message)
                if not isinstance(payload, dict):
                    continue
                if payload.get("command") == "align_simulation_to_leader":
                    try:
                        self._alignment_requests.put_nowait(True)
                    except Full:
                        pass  # Coalesce repeated clicks until the loop consumes one.
                    continue
                self.last_observation = payload
                # print("📥 Observation received:", self.last_observation)
        except websockets.ConnectionClosedOK:
            print("🔌 Connection closed normally.")
        except websockets.ConnectionClosedError as e:
            print(f"⚠️ Connection closed with error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error in handler: {e}")
        finally:
            self._clients.discard(websocket)

    def consume_alignment_request(self):
        """Consume a button click exactly once on the robot-control thread."""
        try:
            self._alignment_requests.get_nowait()
            return True
        except Empty:
            return False

    def send_alignment_result(self, ok: bool, message: str):
        self._schedule_broadcast({
            "type": "leader_alignment_result", "ok": ok, "message": message,
        })

    async def _broadcast_json(self, payload: dict):
        """Send one JSON event to every connected UI client."""
        if not self._clients:
            return

        message = json.dumps(payload)
        clients = tuple(self._clients)
        results = await asyncio.gather(
            *(client.send(message) for client in clients),
            return_exceptions=True,
        )

        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                self._clients.discard(client)

    def _schedule_broadcast(self, payload: dict):
        if not self._loop or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast_json(payload),
            self._loop,
        )

    def send_transform_status(self, status: str):
        """Thread-safe bridge from the robot loop to the WebSocket loop."""
        self._schedule_broadcast({"type": "transform_status", "status": status})

    def send_runtime_status(self, **status):
        """Store and broadcast recording/control state for desktop and XR UIs."""
        self._runtime_status = {**self._runtime_status, **status}
        self._schedule_broadcast({
            "type": "runtime_status",
            **self._runtime_status,
        })

    async def start_websocket_server(self):
        protocol = "wss" if self.ssl_context else "ws"
        print(f"🌐 WebSocket server listening on {protocol}://0.0.0.0:8080")
        self._server = await websockets.serve(
            self.on_observation_received, 
            "0.0.0.0", 
            8080,
            ssl=self.ssl_context
        )
        self._ready_event.set()  # mark as ready once server is up
        if self._stop_requested.is_set():
            self._server.close()
        await self._server.wait_closed()

    def _run_server_thread(self):
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            if not self._stop_requested.is_set():
                self._loop.run_until_complete(self.start_websocket_server())
        except Exception as exc:
            self._startup_error = exc
        finally:
            self.connected = False
            self._ready_event.set()  # Wake connect() on bind/SSL failures too.
            self._loop.close()

    def connect(self):
        """Starts the websocket server in a background thread."""
        print("🚀 Starting websocket server thread...")
        self._ready_event.clear()
        self._stop_requested.clear()
        self._startup_error = None
        self._thread = threading.Thread(target=self._run_server_thread, daemon=True)
        self._thread.start()
        # Wait until the server is ready
        if not self._ready_event.wait(timeout=10.0):
            self.stop()
            raise TimeoutError("WebSocket server did not start within 10 seconds.")
        if self._startup_error is not None:
            self.stop()
            raise RuntimeError("WebSocket server startup failed.") from self._startup_error
        self.connected = True
        protocol = "wss" if self.ssl_context else "ws"
        print(f"✅ WebSocket server is ready and running in background ({protocol}://)")

    def stop(self, timeout: float = 5.0):
        """Close clients/listener on their owning loop, then join its thread."""
        self._stop_requested.set()
        if self._server and self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._server.close)
            except RuntimeError:
                pass  # The loop already finished while shutdown was requested.
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise RuntimeError("WebSocket shutdown did not finish in time.")
        self.connected = False

    async def disconnect(self):
        await asyncio.to_thread(self.stop)


def setup_websocket_server() -> WebSocketServer:
    """Create and connect the WebSocket teleop server."""
    teleop_device = WebSocketServer()
    try:
        teleop_device.connect()
    except BaseException:
        teleop_device.stop()
        raise
    return teleop_device
