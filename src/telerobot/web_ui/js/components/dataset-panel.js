/**
 * Dataset Panel Component
 * Creates a UI panel for recording and saving dataset episodes
 * with a countdown timer before recording begins.
 */
AFRAME.registerComponent('dataset-panel', {
  schema: {
    width: { type: 'number', default: 0.5 },
    height: { type: 'number', default: 0.65 },
    position: { type: 'vec3', default: { x: 1.5, y: 1.2, z: -1.5 } }
  },

  init: function() {
    this.isCountingDown = false;
    this.isRecording = false;
    this.isCheckpointing = false;
    this.isDeletingEpisode = false;
    this.episodeCount = 0;
    this.deleteConfirmUntil = 0;
    this.deleteConfirmTimer = null;
    this.countdownValue = 0;
    this.countdownTimer = null;
    this.datasetConfigured = !!(window.telerobotConfig && window.telerobotConfig.datasetConfigured);

    // Bind methods
    this.onRecordButtonAction = this.onRecordButtonAction.bind(this);
    this.onDeletePreviousButtonAction = this.onDeletePreviousButtonAction.bind(this);
    this.onSaveDatasetButtonAction = this.onSaveDatasetButtonAction.bind(this);
    this.onConnectionChange = this.onConnectionChange.bind(this);
    this.onServerMessage = this.onServerMessage.bind(this);

    this.createPanel();

    // Subscribe to connection state changes
    if (window.webSocketManager) {
      window.webSocketManager.addConnectionChangeListener(this.onConnectionChange);
      window.webSocketManager.addMessageListener(this.onServerMessage);
    }
  },

  createPanel: function() {
    const { width, height, position } = this.data;

    // Main panel container
    this.panel = document.createElement('a-box');
    this.panel.setAttribute('class', 'draggable');
    this.panel.setAttribute('position', `${position.x} ${position.y} ${position.z}`);
    this.panel.setAttribute('width', width);
    this.panel.setAttribute('height', height);
    this.panel.setAttribute('depth', '0.02');
    this.panel.setAttribute('color', '#1a1a2e');
    this.panel.setAttribute('opacity', '0.9');
    this.panel.setAttribute('grabbable', 'startButtons: triggerdown; endButtons: triggerup');
    this.panel.setAttribute('draggable', '');
    this.panel.setAttribute('look-at-headset', 'smoothing: 0.05');

    // Title
    const title = document.createElement('a-text');
    title.setAttribute('value', 'Dataset');
    title.setAttribute('align', 'center');
    title.setAttribute('position', `0 ${height * 0.38} 0.015`);
    title.setAttribute('width', width * 1.5);
    title.setAttribute('color', '#ffffff');
    this.panel.appendChild(title);

    // Keep the headset UI in parity with the desktop/leader UI by showing
    // the saved episode count from the shared runtime status.
    this.episodeText = document.createElement('a-text');
    this.episodeText.setAttribute('value', 'Saved episodes: 0');
    this.episodeText.setAttribute('align', 'center');
    this.episodeText.setAttribute('position', `0 ${height * 0.22} 0.015`);
    this.episodeText.setAttribute('width', width * 2.2);
    this.episodeText.setAttribute('color', '#B0BEC5');
    this.panel.appendChild(this.episodeText);

    // Status / countdown text (always visible)
    this.countdownText = document.createElement('a-text');
    this.countdownText.setAttribute('value', this.datasetConfigured ? 'not recording' : 'no dataset configured');
    this.countdownText.setAttribute('align', 'center');
    this.countdownText.setAttribute('position', `0 ${height * 0.07} 0.015`);
    this.countdownText.setAttribute('width', width * 2.5);
    this.countdownText.setAttribute('color', this.datasetConfigured ? '#FFD54F' : '#EF9A9A');
    this.panel.appendChild(this.countdownText);

    // Record / Save button
    this.recordButton = document.createElement('a-entity');
    this.recordButton.setAttribute('vr-button', {
      width: width * 0.7,
      height: height * 0.12,
      color: '#4CAF50',
      hoverColor: '#66BB6A',
      pressedColor: '#2E7D32',
      text: 'Record Episode',
      textWidth: width * 1.5,
      disabled: true
    });
    this.recordButton.setAttribute('position', `0 ${-height * 0.10} 0.015`);

    this.recordButton.addEventListener('button-action', this.onRecordButtonAction);

    this.panel.appendChild(this.recordButton);

    // Delete previous saved episode button. A two-press confirmation is used in
    // VR because browser confirm() dialogs are awkward/unreliable inside WebXR.
    this.deletePreviousButton = document.createElement('a-entity');
    this.deletePreviousButton.setAttribute('vr-button', {
      width: width * 0.7,
      height: height * 0.12,
      color: '#D32F2F',
      hoverColor: '#EF5350',
      pressedColor: '#B71C1C',
      text: 'Delete Previous',
      textWidth: width * 1.5,
      disabled: true
    });
    this.deletePreviousButton.setAttribute('position', `0 ${-height * 0.25} 0.015`);
    this.deletePreviousButton.addEventListener('button-action', this.onDeletePreviousButtonAction);
    this.panel.appendChild(this.deletePreviousButton);

    // Save Dataset button
    this.saveDatasetButton = document.createElement('a-entity');
    this.saveDatasetButton.setAttribute('vr-button', {
      width: width * 0.7,
      height: height * 0.12,
      color: '#2196F3',
      hoverColor: '#42A5F5',
      pressedColor: '#1565C0',
      text: 'Save Checkpoint',
      textWidth: width * 1.5,
      disabled: true
    });
    this.saveDatasetButton.setAttribute('position', `0 ${-height * 0.40} 0.015`);

    this.saveDatasetButton.addEventListener('button-action', this.onSaveDatasetButtonAction);

    this.panel.appendChild(this.saveDatasetButton);
    this.el.appendChild(this.panel);
  },

  onRecordButtonAction: async function() {
    if (this.isCountingDown) return;

    if (this.isRecording) {
      // Stop episode
      this.isRecording = false;

      if (window.webSocketManager && window.webSocketManager.isConnected) {
        try {
          await window.webSocketManager.triggerAction('stop_episode');
        } catch (error) {
          console.error('❌ stop_episode error:', error);
        }
      }

      console.log('💾 Episode saved');
      this.countdownText.setAttribute('value', 'not recording');

      const buttonComponent = this.recordButton.components['vr-button'];
      if (buttonComponent) {
        buttonComponent.setColor('#4CAF50');
        buttonComponent.setText('Record Episode');
      }
      return;
    }

    // Start countdown
    this.startCountdown();
  },

  startCountdown: function() {
    this.isCountingDown = true;
    this.countdownValue = 3;

    // Disable button during countdown
    const buttonComponent = this.recordButton.components['vr-button'];
    if (buttonComponent) {
      buttonComponent.setDisabled(true);
      buttonComponent.setText('Record Episode');
    }

    // Show countdown value
    this.countdownText.setAttribute('value', String(this.countdownValue));

    this.countdownTimer = setInterval(() => {
      this.countdownValue--;

      if (this.countdownValue > 0) {
        this.countdownText.setAttribute('value', String(this.countdownValue));
      } else {
        // Countdown finished
        clearInterval(this.countdownTimer);
        this.countdownTimer = null;
        this.isCountingDown = false;
        this.isRecording = true;

        // Show recording status
        this.countdownText.setAttribute('value', 'recording');

        // Trigger start_episode action
        if (window.webSocketManager && window.webSocketManager.isConnected) {
          window.webSocketManager.triggerAction('start_episode').catch((error) => {
            console.error('❌ start_episode error:', error);
          });
        }

        // Re-enable button and switch to "Save Episode"
        if (buttonComponent) {
          buttonComponent.setDisabled(false);
          buttonComponent.setColor('#f44336');
          buttonComponent.setText('Save Episode');
        }

        console.log('🔴 Recording started');
      }
    }, 1000);
  },

  onServerMessage: function(message) {
    if (!message || message.type !== 'runtime_status') return;

    if (typeof message.recording === 'boolean') {
      this.isRecording = message.recording;
    }
    if (typeof message.checkpointing === 'boolean') {
      this.isCheckpointing = message.checkpointing;
    }
    if (typeof message.deleting_episode === 'boolean') {
      this.isDeletingEpisode = message.deleting_episode;
    }
    if (typeof message.episode_count === 'number') {
      this.episodeCount = message.episode_count;
      this.episodeText.setAttribute('value', `Saved episodes: ${this.episodeCount}`);
    }

    const isConnected = !!(window.webSocketManager && window.webSocketManager.isConnected);
    const canUseDataset = isConnected
      && this.datasetConfigured
      && !this.isCheckpointing
      && !this.isDeletingEpisode;

    const recordBtnComp = this.recordButton.components['vr-button'];
    if (recordBtnComp && !this.isCountingDown) {
      recordBtnComp.setDisabled(!canUseDataset);
      if (!this.isRecording) {
        recordBtnComp.setColor('#4CAF50');
        recordBtnComp.setText('Record Episode');
      }
    }

    const deleteBtnComp = this.deletePreviousButton.components['vr-button'];
    if (deleteBtnComp) {
      deleteBtnComp.setDisabled(!canUseDataset || this.isRecording || this.episodeCount <= 0);
      if (this.isDeletingEpisode) {
        deleteBtnComp.setText('Deleting...');
      } else if (Date.now() >= this.deleteConfirmUntil) {
        deleteBtnComp.setText('Delete Previous');
      }
    }

    const saveBtnComp = this.saveDatasetButton.components['vr-button'];
    if (saveBtnComp) {
      saveBtnComp.setDisabled(!canUseDataset || this.isRecording);
      saveBtnComp.setText(this.isCheckpointing ? 'Saving...' : 'Save Checkpoint');
    }

    if (this.isCheckpointing) {
      this.countdownText.setAttribute('value', 'saving checkpoint...');
    } else if (this.isDeletingEpisode) {
      this.countdownText.setAttribute('value', 'deleting previous episode...');
    } else if (!this.isRecording && !this.isCountingDown) {
      if (message.delete_episode_error) {
        this.countdownText.setAttribute('value', 'delete failed / Hub retry needed');
      } else {
        this.countdownText.setAttribute('value', message.checkpoint_error ? 'checkpoint failed' : 'not recording');
      }
    }
  },

  onConnectionChange: function(status) {
    const isConnected = status === 'connected';
    const canUseDataset = isConnected
      && this.datasetConfigured
      && !this.isCheckpointing
      && !this.isDeletingEpisode;

    const recordBtnComp = this.recordButton.components['vr-button'];
    if (recordBtnComp && !this.isCountingDown) {
      recordBtnComp.setDisabled(!canUseDataset);
    }

    const deleteBtnComp = this.deletePreviousButton.components['vr-button'];
    if (deleteBtnComp) {
      deleteBtnComp.setDisabled(!canUseDataset || this.isRecording || this.episodeCount <= 0);
    }

    const saveBtnComp = this.saveDatasetButton.components['vr-button'];
    if (saveBtnComp) {
      saveBtnComp.setDisabled(!canUseDataset || this.isRecording);
    }
  },

  onDeletePreviousButtonAction: async function() {
    if (!window.webSocketManager || !window.webSocketManager.isConnected) return;
    if (this.isRecording || this.isCheckpointing || this.isDeletingEpisode || this.episodeCount <= 0) return;

    const buttonComponent = this.deletePreviousButton.components['vr-button'];
    const now = Date.now();

    // First press arms the destructive action for three seconds; second press
    // actually deletes the latest saved episode.
    if (now >= this.deleteConfirmUntil) {
      this.deleteConfirmUntil = now + 3000;
      if (buttonComponent) buttonComponent.setText('Press Again Delete');
      this.countdownText.setAttribute('value', `delete episode ${this.episodeCount - 1}? press again`);
      if (this.deleteConfirmTimer) clearTimeout(this.deleteConfirmTimer);
      this.deleteConfirmTimer = setTimeout(() => {
        this.deleteConfirmUntil = 0;
        if (buttonComponent && !this.isDeletingEpisode) buttonComponent.setText('Delete Previous');
        if (!this.isRecording && !this.isCheckpointing && !this.isDeletingEpisode) {
          this.countdownText.setAttribute('value', 'not recording');
        }
      }, 3000);
      return;
    }

    this.deleteConfirmUntil = 0;
    if (this.deleteConfirmTimer) {
      clearTimeout(this.deleteConfirmTimer);
      this.deleteConfirmTimer = null;
    }
    if (buttonComponent) {
      buttonComponent.setDisabled(true);
      buttonComponent.setText('Deleting...');
    }

    try {
      await window.webSocketManager.triggerAction('delete_previous_episode', 800);
      console.log('🗑️ Delete previous episode requested');
    } catch (error) {
      console.error('❌ delete_previous_episode error:', error);
      if (buttonComponent) {
        buttonComponent.setDisabled(false);
        buttonComponent.setText('Delete Previous');
      }
    }
  },

  onSaveDatasetButtonAction: async function() {
    if (!window.webSocketManager || !window.webSocketManager.isConnected) {
      console.warn('⚠️ WebSocket not connected');
      return;
    }
    if (this.isRecording) {
      console.warn('⚠️ Save the active episode before checkpointing');
      return;
    }
    if (this.isCheckpointing) {
      console.warn('⏳ Dataset checkpoint already in progress');
      return;
    }
    if (this.isDeletingEpisode) {
      console.warn('⏳ Previous episode deletion already in progress');
      return;
    }

    const buttonComponent = this.saveDatasetButton.components['vr-button'];
    if (buttonComponent) {
      buttonComponent.setDisabled(true);
      buttonComponent.setText('...');
    }

    try {
      await window.webSocketManager.triggerAction('save_dataset');
      console.log('💾 Dataset checkpoint requested');
    } catch (error) {
      console.error('❌ save_dataset error:', error);
    } finally {
      if (buttonComponent) {
        buttonComponent.setDisabled(false);
        buttonComponent.setText(this.isCheckpointing ? 'Saving...' : 'Save Checkpoint');
      }
    }
  },

  remove: function() {
    if (this.countdownTimer) {
      clearInterval(this.countdownTimer);
    }
    if (this.deleteConfirmTimer) {
      clearTimeout(this.deleteConfirmTimer);
    }

    if (window.webSocketManager) {
      window.webSocketManager.removeConnectionChangeListener(this.onConnectionChange);
      window.webSocketManager.removeMessageListener(this.onServerMessage);
    }
    this.recordButton.removeEventListener('button-action', this.onRecordButtonAction);
    this.deletePreviousButton.removeEventListener('button-action', this.onDeletePreviousButtonAction);
    this.saveDatasetButton.removeEventListener('button-action', this.onSaveDatasetButtonAction);

    if (this.panel && this.panel.parentNode) {
      this.panel.parentNode.removeChild(this.panel);
    }
  }
});
