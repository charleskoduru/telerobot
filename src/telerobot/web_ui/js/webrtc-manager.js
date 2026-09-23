/**
 * WebRTC Connection Manager
 * Handles WebRTC connections to camera streams from the server
 */
const WebRTCManager = {
  serverUrl: window.location.origin,
  connections: {},

  /**
   * Fetch the list of available cameras from the server
   * @returns {Promise<string[]>} Array of camera names
   */
  async getCameras() {
    try {
      const response = await fetch(`${this.serverUrl}/cameras`);
      if (!response.ok) throw new Error(`Camera list: HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch cameras:', error);
      return [];
    }
  },

  /**
   * Connect to a camera stream via WebRTC
   * @param {string} cameraName - The name of the camera to connect to
   * @param {HTMLVideoElement} videoElement - The video element to stream to
   * @returns {Promise<boolean>} True if connection was successful
   */
  async connectToCamera(cameraName, videoElement) {
    let pc;
    try {
      this.disconnect(cameraName);
      pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      });

      pc.ontrack = (event) => {
        console.log(`Received track for camera: ${cameraName}`);
        videoElement.srcObject = event.streams[0] || new MediaStream([event.track]);
        videoElement.play().catch(e => console.log('Autoplay prevented:', e));
      };

      pc.oniceconnectionstatechange = () => {
        console.log(`ICE state for ${cameraName}: ${pc.iceConnectionState}`);
      };

      // Create offer
      pc.addTransceiver('video', { direction: 'recvonly' });
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      // /offer has no trickle-ICE endpoint; send the SDP after local ICE
      // candidates have been collected (important on the Quest/local Wi-Fi).
      if (pc.iceGatheringState !== 'complete') {
        await new Promise(resolve => {
          const timeout = setTimeout(done, 4000);
          function done() {
            clearTimeout(timeout);
            pc.removeEventListener('icegatheringstatechange', check);
            resolve();
          }
          function check() {
            if (pc.iceGatheringState === 'complete') done();
          }
          pc.addEventListener('icegatheringstatechange', check);
          check();
        });
      }

      // Send offer to server
      const response = await fetch(`${this.serverUrl}/offer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
          camera: cameraName
        })
      });

      if (!response.ok) throw new Error(`Camera ${cameraName}: HTTP ${response.status}`);
      const answer = await response.json();
      await pc.setRemoteDescription(new RTCSessionDescription(answer));

      this.connections[cameraName] = pc;
      console.log(`Connected to camera: ${cameraName}`);
      return true;
    } catch (error) {
      if (pc) pc.close();
      console.error(`Failed to connect to camera ${cameraName}:`, error);
      return false;
    }
  },

  /**
   * Disconnect from a camera stream
   * @param {string} cameraName - The name of the camera to disconnect from
   */
  disconnect(cameraName) {
    if (this.connections[cameraName]) {
      this.connections[cameraName].close();
      delete this.connections[cameraName];
      console.log(`Disconnected from camera: ${cameraName}`);
    }
  },

  /**
   * Disconnect from all camera streams
   */
  disconnectAll() {
    Object.keys(this.connections).forEach(cameraName => {
      this.disconnect(cameraName);
    });
  }
};

// Export for use in other modules
window.WebRTCManager = WebRTCManager;