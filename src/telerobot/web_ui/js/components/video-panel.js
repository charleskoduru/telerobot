/**
 * Video Panel Component
 * Creates a single unified panel containing all camera streams
 * Layout rules:
 * - 1 camera: fills the panel
 * - 2 cameras: gripperCam on top, bevCam below, equal size
 * - 3+ cameras: main on top (full width), others split the bottom row
 */
AFRAME.registerComponent('video-panel', {
  schema: {
    radius: { type: 'number', default: 2 },        // Distance from user
    height: { type: 'number', default: 1.5 },      // Height of panel center
    maxWidth: { type: 'number', default: 1.6 },    // Max panel width
    maxHeight: { type: 'number', default: 1.8 },   // Max panel height
    smoothing: { type: 'number', default: 0.08 },  // Look-at smoothing
    padding: { type: 'number', default: 0.02 },    // Padding between streams
    mainCameraName: { type: 'string', default: 'gripperCam' }
  },

  init: async function() {
    this.panel = null;
    this.videoStreams = [];
    this.removed = false;

    // The desktop dashboard owns its own HTML video elements and WebRTC peers.
    if (window.telerobotClientMode === 'desktop') return;
    
    // Wait for scene to be fully loaded
    if (this.el.sceneEl.hasLoaded) {
      await this.createUnifiedPanel();
    } else {
      this.el.sceneEl.addEventListener('loaded', () => this.createUnifiedPanel());
    }
  },

  createUnifiedPanel: async function() {
    const cameras = await WebRTCManager.getCameras();
    console.log('Available cameras:', cameras);

    if (cameras.length === 0) {
      console.log('No cameras available');
      this.createPlaceholderPanel();
      return;
    }

    // Sort cameras: main first, then others
    const sortedCameras = this.sortCameras(cameras);

    // Create the main panel container
    this.createPanelContainer();

    // Create video streams for each camera
    sortedCameras.forEach((cameraName, index) => {
      this.createVideoStream(cameraName, index);
    });
    // Draw both tiles immediately. A slow or disconnected camera must not hide
    // the other camera; each tile becomes live when its own video is ready.
    this.layoutStreams();
    this.videoStreams.forEach(stream => {
      if (stream.videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        this.onStreamReady(stream);
      }
    });
  },

  /**
   * Sort cameras: "main" first, then cameras containing "left", then others
   */
  sortCameras: function(cameras) {
    const mainName = this.data.mainCameraName.toLowerCase();
    return [...cameras].sort((a, b) => {
      const aLower = a.toLowerCase();
      const bLower = b.toLowerCase();
      
      const aIsMain = aLower === mainName;
      const bIsMain = bLower === mainName;
      const aIsLeft = aLower.includes('left');
      const bIsLeft = bLower.includes('left');
      
      // Main camera comes first
      if (aIsMain && !bIsMain) return -1;
      if (!aIsMain && bIsMain) return 1;
      
      // Then cameras containing "left"
      if (aIsLeft && !bIsLeft) return -1;
      if (!aIsLeft && bIsLeft) return 1;
      
      return 0;
    });
  },

  createPlaceholderPanel: function() {
    const { radius, height, maxWidth, maxHeight, smoothing } = this.data;

    const panel = document.createElement('a-box');
    panel.setAttribute('class', 'draggable');
    panel.setAttribute('position', `0 ${height} ${-radius}`);
    panel.setAttribute('width', maxWidth);
    panel.setAttribute('height', maxHeight / 2);
    panel.setAttribute('depth', '0.05');
    panel.setAttribute('color', '#333');
    panel.setAttribute('grabbable', 'startButtons: triggerdown; endButtons: triggerup');
    panel.setAttribute('draggable', '');
    panel.setAttribute('shadow', '');
    panel.setAttribute('look-at-headset', `smoothing: ${smoothing}`);

    const text = document.createElement('a-text');
    text.setAttribute('value', 'No cameras available');
    text.setAttribute('align', 'center');
    text.setAttribute('position', '0 0 0.03');
    text.setAttribute('width', '2');
    text.setAttribute('color', '#ffffff');
    panel.appendChild(text);

    this.el.appendChild(panel);
  },

  createPanelContainer: function() {
    const { radius, height, maxWidth, maxHeight, smoothing } = this.data;

    // Create the unified panel as a box (needs geometry for raycaster to hit)
    this.panel = document.createElement('a-box');
    this.panel.setAttribute('class', 'draggable');
    this.panel.setAttribute('position', `0 ${height} ${-radius}`);
    this.panel.setAttribute('width', maxWidth);
    this.panel.setAttribute('height', maxHeight);
    this.panel.setAttribute('depth', '0.05');
    this.panel.setAttribute('color', '#222');
    this.panel.setAttribute('shadow', '');
    this.panel.setAttribute('grabbable', 'startButtons: triggerdown; endButtons: triggerup');
    this.panel.setAttribute('draggable', '');
    this.panel.setAttribute('look-at-headset', `smoothing: ${smoothing}`);

    this.el.appendChild(this.panel);
  },

  /**
   * Create a video stream element
   */
  createVideoStream: function(cameraName, index) {
    const streamData = {
      cameraName,
      index,
      videoEl: null,
      planeEl: null,
      textEl: null,
      ready: false,
      width: 640,
      height: 480
    };

    this.videoStreams.push(streamData);

    // A-Frame can use this same video element directly as its texture.
    const videoEl = document.createElement('video');
    videoEl.id = `video-stream-${index}`;
    videoEl.setAttribute('playsinline', '');
    videoEl.setAttribute('autoplay', '');
    videoEl.muted = true;
    const assets = document.querySelector('a-assets') || this.createAssets();
    assets.appendChild(videoEl);
    streamData.videoEl = videoEl;

    videoEl.addEventListener('loadeddata', () => this.onStreamReady(streamData));

    console.log(`Creating stream ${index} for camera: ${cameraName}`);

    // Connect to WebRTC stream
    this.connectToStream(streamData);
  },

  connectToStream: async function(streamData) {
    const connected = await WebRTCManager.connectToCamera(streamData.cameraName, streamData.videoEl);

    if (!connected && !this.removed && streamData.textEl) {
      streamData.textEl.setAttribute('value', `${streamData.cameraName}\nFailed to connect`);
    } else if (streamData.videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      this.onStreamReady(streamData);
    }
  },

  onStreamReady: function(streamData) {
    if (this.removed || streamData.ready || !streamData.planeEl) return;
    streamData.ready = true;

    // Get video dimensions
    streamData.width = streamData.videoEl.videoWidth || 640;
    streamData.height = streamData.videoEl.videoHeight || 480;

    console.log(`Stream ${streamData.index} (${streamData.cameraName}) ready: ${streamData.width}x${streamData.height}`);

    // Video is display color data. Mark the texture as sRGB and keep it out of
    // renderer tone mapping so the Quest does not make the feed look darker.
    const applyVideoColorSettings = () => {
      const mesh = streamData.planeEl.getObject3D('mesh');
      const material = mesh && mesh.material;
      const texture = material && material.map;

      if (!material || !texture) return;

      material.toneMapped = false;
      if ('colorSpace' in texture && THREE.SRGBColorSpace) {
        texture.colorSpace = THREE.SRGBColorSpace;
      } else if ('encoding' in texture && THREE.sRGBEncoding) {
        texture.encoding = THREE.sRGBEncoding;
      }
      texture.needsUpdate = true;
      material.needsUpdate = true;
    };

    streamData.planeEl.addEventListener('materialtextureloaded', applyVideoColorSettings);
    streamData.planeEl.addEventListener('materialvideoloadeddata', applyVideoColorSettings);
    streamData.planeEl.setAttribute(
      'material',
      `shader: flat; src: #${streamData.videoEl.id}; side: front; color: #FFFFFF; toneMapped: false; fog: false`
    );
    requestAnimationFrame(applyVideoColorSettings);
    setTimeout(applyVideoColorSettings, 500);
    if (streamData.textEl && streamData.textEl.parentNode) {
      streamData.textEl.parentNode.removeChild(streamData.textEl);
    }
  },

  /**
   * Layout all streams within the panel based on the rules
   */
  layoutStreams: function() {
    const { maxWidth, maxHeight, padding } = this.data;
    const numStreams = this.videoStreams.length;

    if (numStreams === 0) return;

    // Get the main stream (first one after sorting)
    const mainStream = this.videoStreams[0];
    const mainAspect = mainStream.width / mainStream.height;

    let panelWidth, panelHeight;
    let layouts = [];

    if (numStreams === 1) {
      // Single camera: fill the panel
      if (mainAspect > maxWidth / maxHeight) {
        panelWidth = maxWidth;
        panelHeight = maxWidth / mainAspect;
      } else {
        panelHeight = maxHeight;
        panelWidth = maxHeight * mainAspect;
      }

      layouts.push({
        stream: mainStream,
        x: 0,
        y: 0,
        width: panelWidth - padding * 2,
        height: panelHeight - padding * 2
      });

    } else if (numStreams === 2) {
      // Two cameras: main on top, other below, equal size
      const streamHeight = (maxHeight - padding * 3) / 2;
      
      // Calculate width based on main camera aspect ratio
      panelWidth = Math.min(maxWidth, streamHeight * mainAspect + padding * 2);
      panelHeight = maxHeight;
      
      const streamWidth = panelWidth - padding * 2;
      const actualStreamHeight = streamWidth / mainAspect;
      
      // Recalculate panel height to fit both streams
      panelHeight = actualStreamHeight * 2 + padding * 3;
      
      // Clamp to maxHeight
      if (panelHeight > maxHeight) {
        panelHeight = maxHeight;
        const availableHeight = (panelHeight - padding * 3) / 2;
        panelWidth = Math.min(maxWidth, availableHeight * mainAspect + padding * 2);
      }
      
      const finalStreamWidth = panelWidth - padding * 2;
      const finalStreamHeight = (panelHeight - padding * 3) / 2;

      // Main on top
      layouts.push({
        stream: mainStream,
        x: 0,
        y: finalStreamHeight / 2 + padding / 2,
        width: finalStreamWidth,
        height: finalStreamHeight
      });

      // Other below
      layouts.push({
        stream: this.videoStreams[1],
        x: 0,
        y: -(finalStreamHeight / 2 + padding / 2),
        width: finalStreamWidth,
        height: finalStreamHeight
      });

    } else {
      // 3+ cameras: main on top (full width), others split the bottom row
      const otherStreams = this.videoStreams.slice(1);
      const numBottom = otherStreams.length;

      // Top stream takes roughly 60% of height, bottom row takes 40%
      const topHeightRatio = 0.6;
      const bottomHeightRatio = 0.4;

      // Calculate based on main camera aspect
      panelWidth = Math.min(maxWidth, (maxHeight * topHeightRatio) * mainAspect);
      panelHeight = maxHeight;

      const topHeight = (panelHeight - padding * 3) * topHeightRatio;
      const bottomHeight = (panelHeight - padding * 3) * bottomHeightRatio;
      const topWidth = panelWidth - padding * 2;

      // Main on top
      layouts.push({
        stream: mainStream,
        x: 0,
        y: bottomHeight / 2 + padding / 2,
        width: topWidth,
        height: topHeight
      });

      // Others on bottom, split evenly
      const bottomStreamWidth = (panelWidth - padding * (numBottom + 1)) / numBottom;
      const totalBottomWidth = bottomStreamWidth * numBottom + padding * (numBottom - 1);
      const startX = -totalBottomWidth / 2 + bottomStreamWidth / 2;

      otherStreams.forEach((stream, idx) => {
        layouts.push({
          stream: stream,
          x: startX + idx * (bottomStreamWidth + padding),
          y: -(topHeight / 2 + padding / 2),
          width: bottomStreamWidth,
          height: bottomHeight
        });
      });
    }

    // Update panel size (panel is the box itself)
    this.panel.setAttribute('width', panelWidth);
    this.panel.setAttribute('height', panelHeight);

    // Create video planes for each layout
    layouts.forEach(layout => this.createVideoPlane(layout));

    console.log(`Panel laid out with ${numStreams} streams: ${panelWidth.toFixed(2)}x${panelHeight.toFixed(2)}`);
  },

  createVideoPlane: function(layout) {
    const { stream, x, y, width, height } = layout;

    // The texture is attached only after loadeddata; until then the tile
    // visibly reports which camera is still connecting.
    const planeEl = document.createElement('a-plane');
    planeEl.setAttribute('position', `${x} ${y} 0.040`);
    planeEl.setAttribute('width', width);
    planeEl.setAttribute('height', height);
    planeEl.setAttribute('material', 'shader: flat; color: #222222; side: front');
    this.panel.appendChild(planeEl);
    stream.planeEl = planeEl;

    const textEl = document.createElement('a-text');
    textEl.setAttribute('value', `${stream.cameraName}\nConnecting...`);
    textEl.setAttribute('align', 'center');
    textEl.setAttribute('position', `${x} ${y} 0.046`);
    textEl.setAttribute('width', Math.min(width, 1.5));
    textEl.setAttribute('color', '#ffffff');
    this.panel.appendChild(textEl);
    stream.textEl = textEl;

    // Add label at the top of this stream with semi-transparent background
    const labelHeight = 0.08;
    const labelY = y + height / 2 - labelHeight / 2;
    
    // Estimate text width based on character count (approx 0.05 per character at width 1.2)
    const textPadding = 0.04;
    const charWidth = 0.045;
    const labelTextWidth = stream.cameraName.length * charWidth + textPadding * 2;
    
    // Semi-transparent background for label (sized to text)
    const labelBgEl = document.createElement('a-plane');
    labelBgEl.setAttribute('position', `${x} ${labelY} 0.043`);
    labelBgEl.setAttribute('width', labelTextWidth);
    labelBgEl.setAttribute('height', labelHeight);
    labelBgEl.setAttribute('material', 'shader: flat; color: #000000; opacity: 0.5; transparent: true');
    this.panel.appendChild(labelBgEl);
    
    // Label text
    const labelEl = document.createElement('a-text');
    labelEl.setAttribute('value', stream.cameraName);
    labelEl.setAttribute('align', 'center');
    labelEl.setAttribute('position', `${x} ${labelY} 0.046`);
    labelEl.setAttribute('width', '1.2');
    labelEl.setAttribute('color', '#ffffff');
    this.panel.appendChild(labelEl);

    console.log(`Created video plane for ${stream.cameraName} at (${x.toFixed(2)}, ${y.toFixed(2)}) size ${width.toFixed(2)}x${height.toFixed(2)}`);
  },

  createAssets: function() {
    const assets = document.createElement('a-assets');
    this.el.sceneEl.insertBefore(assets, this.el.sceneEl.firstChild);
    return assets;
  },

  /**
   * Remove all panels
   */
  clearPanels: function() {
    // Release the old peer connections and textures before re-creating tiles.
    this.videoStreams.forEach(stream => {
      WebRTCManager.disconnect(stream.cameraName);
      if (stream.videoEl && stream.videoEl.parentNode) {
        stream.videoEl.parentNode.removeChild(stream.videoEl);
      }
    });
    this.videoStreams = [];

    while (this.el.firstChild) {
      this.el.removeChild(this.el.firstChild);
    }
    this.panel = null;
  },

  /**
   * Refresh panels (re-fetch cameras and recreate)
   */
  refresh: async function() {
    this.clearPanels();
    await this.createUnifiedPanel();
  },

  remove: function() {
    this.removed = true;
    this.clearPanels();
  }
});
