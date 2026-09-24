/**
 * Minimal headset status for A-button controller-to-end-effector recalibration.
 */
AFRAME.registerComponent('transform-status', {
  schema: {
    position: { type: 'vec3', default: { x: 0, y: 0.43, z: -1.5 } }
  },

  init: function() {
    this.state = 'offline';
    this.resetTimer = null;

    this.onLocalStatus = this.onLocalStatus.bind(this);
    this.onServerMessage = this.onServerMessage.bind(this);
    this.onConnectionChange = this.onConnectionChange.bind(this);

    this.createPanel();
    window.addEventListener('transform-status-local', this.onLocalStatus);

    if (window.webSocketManager) {
      window.webSocketManager.addMessageListener(this.onServerMessage);
      window.webSocketManager.addConnectionChangeListener(this.onConnectionChange);
      this.setState(window.webSocketManager.isConnected ? 'idle' : 'offline');
    }
  },

  createPanel: function() {
    const position = this.data.position;

    this.panel = document.createElement('a-box');
    this.panel.setAttribute('class', 'draggable');
    this.panel.setAttribute('position', `${position.x} ${position.y} ${position.z}`);
    this.panel.setAttribute('width', '0.72');
    this.panel.setAttribute('height', '0.16');
    this.panel.setAttribute('depth', '0.012');
    this.panel.setAttribute('color', '#10141C');
    this.panel.setAttribute('opacity', '0.92');
    this.panel.setAttribute('grabbable', 'startButtons: triggerdown; endButtons: triggerup');
    this.panel.setAttribute('draggable', '');
    this.panel.setAttribute('look-at-headset', 'smoothing: 0.05');

    const badge = document.createElement('a-circle');
    badge.setAttribute('radius', '0.045');
    badge.setAttribute('position', '-0.29 0 0.013');
    badge.setAttribute('color', '#E8EDF5');
    this.panel.appendChild(badge);

    const badgeText = document.createElement('a-text');
    badgeText.setAttribute('value', 'A');
    badgeText.setAttribute('align', 'center');
    badgeText.setAttribute('anchor', 'center');
    badgeText.setAttribute('position', '-0.29 -0.014 0.014');
    badgeText.setAttribute('width', '0.26');
    badgeText.setAttribute('color', '#10141C');
    this.panel.appendChild(badgeText);

    const title = document.createElement('a-text');
    title.setAttribute('value', 'Reset transform');
    title.setAttribute('align', 'left');
    title.setAttribute('anchor', 'left');
    title.setAttribute('position', '-0.215 0.025 0.014');
    title.setAttribute('width', '0.78');
    title.setAttribute('color', '#FFFFFF');
    this.panel.appendChild(title);

    this.statusText = document.createElement('a-text');
    this.statusText.setAttribute('align', 'left');
    this.statusText.setAttribute('anchor', 'left');
    this.statusText.setAttribute('position', '-0.215 -0.035 0.014');
    this.statusText.setAttribute('width', '0.66');
    this.statusText.setAttribute('color', '#9AA6B2');
    this.panel.appendChild(this.statusText);

    this.statusDot = document.createElement('a-circle');
    this.statusDot.setAttribute('radius', '0.012');
    this.statusDot.setAttribute('position', '0.31 0 0.014');
    this.panel.appendChild(this.statusDot);

    this.el.appendChild(this.panel);
  },

  onLocalStatus: function(event) {
    if (event.detail && event.detail.status) {
      this.setState(event.detail.status);
    }
  },

  onServerMessage: function(message) {
    if (message.type === 'transform_status') {
      this.setState(message.status);
    }
  },

  onConnectionChange: function(status) {
    if (status === 'connected') {
      if (this.state === 'offline' || this.state === 'error') this.setState('idle');
    } else if (status === 'disconnected') {
      this.setState('offline');
    }
  },

  setState: function(state) {
    const styles = {
      offline: { label: 'Connect first', color: '#EF6C73' },
      idle: { label: 'Press A', color: '#7E8A98' },
      requesting: { label: 'Starting...', color: '#F4B860' },
      collecting: { label: 'Hold Grip to capture', color: '#F4B860' },
      finished: { label: 'Pose captured', color: '#62D49A' },
      error: { label: 'Try again', color: '#EF6C73' }
    };
    const style = styles[state] || styles.idle;

    this.state = state;
    this.statusText.setAttribute('value', style.label);
    this.statusText.setAttribute('color', style.color);
    this.statusDot.setAttribute('color', style.color);
    this.statusDot.removeAttribute('animation__pulse');
    this.statusDot.setAttribute('scale', '1 1 1');

    if (state === 'requesting' || state === 'collecting') {
      this.statusDot.setAttribute(
        'animation__pulse',
        'property: scale; from: 1 1 1; to: 1.45 1.45 1.45; dir: alternate; loop: true; dur: 450; easing: easeInOutSine'
      );
    }

    if (this.resetTimer) {
      clearTimeout(this.resetTimer);
      this.resetTimer = null;
    }

    if (state === 'finished') {
      this.resetTimer = setTimeout(() => this.setState('idle'), 1800);
    } else if (state === 'error') {
      this.resetTimer = setTimeout(() => {
        const connected = window.webSocketManager && window.webSocketManager.isConnected;
        this.setState(connected ? 'idle' : 'offline');
      }, 1800);
    }
  },

  remove: function() {
    window.removeEventListener('transform-status-local', this.onLocalStatus);
    if (window.webSocketManager) {
      window.webSocketManager.removeMessageListener(this.onServerMessage);
      window.webSocketManager.removeConnectionChangeListener(this.onConnectionChange);
    }
    if (this.resetTimer) clearTimeout(this.resetTimer);
    if (this.panel && this.panel.parentNode) this.panel.parentNode.removeChild(this.panel);
  }
});
