/** Desktop camera and dataset controls. The Quest keeps the existing A-Frame UI. */
class DesktopDashboard {
  constructor() {
    this.config = window.telerobotConfig || {};
    this.connected = false;
    this.countingDown = false;
    this.runtime = {
      control_mode: this.config.teleoperationMode || 'vr',
      dataset_configured: !!this.config.datasetConfigured,
      recording: false,
      finalized: false,
      episode_count: 0
    };

    this.root = document.getElementById('desktop-dashboard');
    this.modeBadge = document.getElementById('desktop-mode-badge');
    this.connectionBadge = document.getElementById('desktop-connection-badge');
    this.recordingStatus = document.getElementById('desktop-recording-status');
    this.recordingDot = document.getElementById('desktop-recording-dot');
    this.episodeCount = document.getElementById('desktop-episode-count');
    this.recordButton = document.getElementById('desktop-record-button');
    this.resetButton = document.getElementById('desktop-reset-button');
    this.finalizeButton = document.getElementById('desktop-finalize-button');
    this.connectButton = document.getElementById('desktop-connect-button');
    this.refreshButton = document.getElementById('desktop-refresh-cameras');
    this.cameraGrid = document.getElementById('desktop-camera-grid');
    this.helpText = document.getElementById('desktop-help');

    this.onConnectionChange = this.onConnectionChange.bind(this);
    this.onServerMessage = this.onServerMessage.bind(this);
  }

  async init() {
    document.body.classList.add('desktop-client');
    this.root.hidden = false;

    this.recordButton.addEventListener('click', () => this.toggleRecording());
    this.resetButton.addEventListener('click', () => this.resetRobot());
    this.finalizeButton.addEventListener('click', () => this.finalizeDataset());
    this.connectButton.addEventListener('click', () => this.connect());
    this.refreshButton.addEventListener('click', () => this.loadCameras());

    window.webSocketManager.addConnectionChangeListener(this.onConnectionChange);
    window.webSocketManager.addMessageListener(this.onServerMessage);

    this.render();
    await Promise.allSettled([this.connect(), this.loadCameras()]);
  }

  async connect() {
    this.helpText.textContent = 'Connecting to the robot control server…';
    try {
      await window.webSocketManager.connect();
    } catch (error) {
      console.error('Desktop WebSocket connection failed:', error);
      this.helpText.textContent = 'Connection failed. Check the certificate, then reconnect.';
    }
  }

  onConnectionChange(status) {
    this.connected = status === WebSocketManager.STATUS_CONNECTED;
    this.helpText.textContent = this.connected
      ? 'Controls are ready.'
      : (status === WebSocketManager.STATUS_CONNECTING ? 'Connecting…' : 'Control server disconnected.');
    this.render();
  }

  onServerMessage(message) {
    if (message.type !== 'runtime_status') return;
    this.runtime = { ...this.runtime, ...message };
    this.render();
  }

  render() {
    const modeLabel = this.runtime.control_mode === 'leader' ? 'LEADER ARM' : 'VR';
    this.modeBadge.textContent = modeLabel;

    this.connectionBadge.textContent = this.connected ? 'Connected' : 'Offline';
    this.connectionBadge.classList.toggle('badge-online', this.connected);
    this.connectionBadge.classList.toggle('badge-offline', !this.connected);

    this.episodeCount.textContent = String(this.runtime.episode_count || 0);
    this.recordingDot.classList.toggle('recording-dot-active', !!this.runtime.recording);

    if (this.countingDown) {
      // Countdown text is managed by startCountdown().
    } else if (this.runtime.finalized) {
      this.recordingStatus.textContent = 'Dataset finalized';
    } else {
      this.recordingStatus.textContent = this.runtime.recording ? 'Recording' : 'Not recording';
    }

    const datasetReady = this.connected
      && this.runtime.dataset_configured
      && !this.runtime.finalized;
    this.recordButton.disabled = !datasetReady || this.countingDown;
    this.recordButton.textContent = this.runtime.recording ? 'Stop & save episode' : 'Start episode';
    this.recordButton.classList.toggle('button-danger', !!this.runtime.recording);
    this.resetButton.disabled = !this.connected;
    this.finalizeButton.disabled = !datasetReady || !!this.runtime.recording || this.countingDown;

    if (!this.runtime.dataset_configured) {
      this.helpText.textContent = 'No dataset is configured in config.yaml.';
    } else if (this.runtime.finalized) {
      this.helpText.textContent = 'Dataset finalized. Restart Telerobot to record more episodes.';
    }
  }

  async loadCameras() {
    this.refreshButton.disabled = true;
    this.cameraGrid.replaceChildren();
    const loading = document.createElement('p');
    loading.className = 'empty-message';
    loading.textContent = 'Loading cameras…';
    this.cameraGrid.appendChild(loading);

    window.WebRTCManager.disconnectAll();
    const cameras = await window.WebRTCManager.getCameras();
    this.cameraGrid.replaceChildren();

    if (!cameras.length) {
      const empty = document.createElement('p');
      empty.className = 'empty-message';
      empty.textContent = 'No cameras available.';
      this.cameraGrid.appendChild(empty);
      this.refreshButton.disabled = false;
      return;
    }

    for (const cameraName of cameras) {
      const figure = document.createElement('figure');
      figure.className = 'camera-card';

      const video = document.createElement('video');
      video.autoplay = true;
      video.muted = true;
      video.playsInline = true;
      video.setAttribute('aria-label', `${cameraName} live camera`);

      const caption = document.createElement('figcaption');
      caption.textContent = cameraName;

      figure.append(video, caption);
      this.cameraGrid.appendChild(figure);

      const connected = await window.WebRTCManager.connectToCamera(cameraName, video);
      if (!connected) figure.classList.add('camera-card-error');
    }

    this.refreshButton.disabled = false;
  }

  async toggleRecording() {
    if (this.runtime.recording) {
      await this.sendAction('stop_episode');
      return;
    }
    await this.startCountdown();
  }

  async startCountdown() {
    if (this.countingDown || !this.connected) return;
    this.countingDown = true;
    this.render();

    for (let count = 3; count > 0; count -= 1) {
      this.recordingStatus.textContent = `Recording in ${count}`;
      await new Promise(resolve => setTimeout(resolve, 1000));
      if (!this.connected) {
        this.countingDown = false;
        this.render();
        return;
      }
    }

    await this.sendAction('start_episode');
    this.countingDown = false;
    this.render();
  }

  async resetRobot() {
    const leaderWarning = this.runtime.control_mode === 'leader'
      ? ' Move the leader arm near the follower reset pose before continuing.'
      : '';
    if (!window.confirm(`Return the robot to its captured initial pose?${leaderWarning}`)) return;
    await this.sendAction('reset', 800);
  }

  async finalizeDataset() {
    if (!window.confirm('Finalize the dataset and upload it if push_to_hub is enabled?')) return;
    await this.sendAction('save_dataset', 800);
  }

  async sendAction(action, durationMs = 250) {
    try {
      await window.webSocketManager.triggerAction(action, durationMs);
    } catch (error) {
      console.error(`Desktop action failed: ${action}`, error);
      this.helpText.textContent = 'Action was not sent. Wait a moment and try again.';
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  if (window.telerobotClientMode !== 'desktop') return;
  const dashboard = new DesktopDashboard();
  window.desktopDashboard = dashboard;
  dashboard.init();
});
