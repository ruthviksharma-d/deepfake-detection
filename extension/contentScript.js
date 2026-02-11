
//exp (with ~85% accuracy)


(async () => {
  const contentScript = {
    overlay: null,
    isAnalyzing: false,
    analysisInterval: null,
    lastAnalysisTime: 0,
    analysisHistory: [],
    currentVideo: null,
    frameQuality: 'medium', // Start with medium quality for speed
    lastFrameHash: null,
    isEnabled: true,
    
    // Optimized Configuration for better performance
    config: {
      analysisIntervalMs: 3000, // Analyze every 3 seconds
      minTimeBetweenAnalysis: 2500, // Minimum time between analyses
      historySize: 5, // Reduced history for less lag
      confidenceThreshold: 0.4, // LOWERED from 0.65 to 0.4 for more decisive results
      consecutiveDetectionThreshold: 2, // Reduced for faster response
      statusDisplayDuration: 2500, // Shorter display duration
      skipIdenticalFrames: true, // Skip identical frames
      adaptiveInterval: true, // Adapt interval based on results
    },

    async initialize() {
      console.log('🛡️ DeepFake Shield Enhanced - Initializing...');
      this.createOptimizedOverlay();
      this.addMessageListener();
      this.startVideoMonitoring();
      this.addKeyboardShortcuts();
      this.setupPerformanceMonitoring();
      console.log('✅ DeepFake Shield initialized successfully');
    },

    createOptimizedOverlay() {
      if (this.overlay) return;
      
      this.overlay = document.createElement('div');
      this.overlay.id = 'deepfake-shield-overlay';
      this.overlay.innerHTML = `
        <div class="status-indicator">
          <div class="status-icon">🛡️</div>
          <div class="status-text">Ready</div>
          <div class="confidence-bar">
            <div class="confidence-fill" style="width: 0%"></div>
          </div>
          <div class="details">
            <span class="score">Score: --</span>
            <span class="timing">Time: --ms</span>
          </div>
        </div>
        <div class="controls">
          <button class="toggle-btn" title="Toggle monitoring">⏸️</button>
          <button class="quality-btn" title="Toggle quality">MD</button>
          <button class="speed-btn" title="Speed mode">🚀</button>
        </div>
      `;
      
      document.body.appendChild(this.overlay);
      this.addOptimizedStyles();
      this.addOverlayEventListeners();
    },

    addOptimizedStyles() {
      const style = document.createElement('style');
      style.textContent = `
        #deepfake-shield-overlay {
          position: fixed;
          top: 20px;
          right: 20px;
          z-index: 10000;
          background: rgba(0, 0, 0, 0.9);
          border-radius: 10px;
          padding: 10px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
          backdrop-filter: blur(8px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          min-width: 180px;
          transition: all 0.2s ease;
          will-change: transform, background-color;
        }

        .status-indicator {
          display: flex;
          flex-direction: column;
          gap: 6px;
          margin-bottom: 6px;
        }

        .status-icon {
          font-size: 16px;
          text-align: center;
        }

        .status-text {
          color: white;
          font-weight: 600;
          font-size: 11px;
          text-align: center;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          transition: color 0.2s ease;
        }

        .confidence-bar {
          height: 3px;
          background: rgba(255, 255, 255, 0.2);
          border-radius: 2px;
          overflow: hidden;
        }

        .confidence-fill {
          height: 100%;
          background: linear-gradient(90deg, #ff4757, #ffa502, #2ed573);
          border-radius: 2px;
          transition: width 0.3s ease;
          will-change: width;
        }

        .details {
          display: flex;
          justify-content: space-between;
          font-size: 9px;
          color: rgba(255, 255, 255, 0.7);
        }

        .controls {
          display: flex;
          gap: 3px;
          justify-content: center;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          padding-top: 6px;
        }

        .controls button {
          background: rgba(255, 255, 255, 0.1);
          border: none;
          border-radius: 4px;
          padding: 3px 6px;
          font-size: 10px;
          color: white;
          cursor: pointer;
          transition: background 0.2s ease;
        }

        .controls button:hover {
          background: rgba(255, 255, 255, 0.2);
        }

        .controls button.active {
          background: rgba(46, 213, 115, 0.3);
        }

        /* Status-specific styles with GPU acceleration */
        .deepfake-detected {
          background: rgba(255, 71, 87, 0.95) !important;
          border-color: #ff4757 !important;
          transform: translateZ(0);
        }

        .real-detected {
          background: rgba(46, 213, 115, 0.95) !important;
          border-color: #2ed573 !important;
          transform: translateZ(0);
        }

        .analyzing {
          background: rgba(255, 165, 2, 0.95) !important;
          border-color: #ffa502 !important;
          transform: translateZ(0);
        }

        /* Reduced animation for performance */
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.8; }
        }

        .pulse {
          animation: pulse 1s ease-in-out;
        }

        /* Performance optimizations */
        * {
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

        /* Hide during fullscreen */
        video:fullscreen ~ #deepfake-shield-overlay,
        video:-webkit-full-screen ~ #deepfake-shield-overlay {
          display: none;
        }
      `;
      document.head.appendChild(style);
    },

    addOverlayEventListeners() {
      const toggleBtn = this.overlay.querySelector('.toggle-btn');
      const qualityBtn = this.overlay.querySelector('.quality-btn');
      const speedBtn = this.overlay.querySelector('.speed-btn');

      toggleBtn?.addEventListener('click', () => {
        this.toggleMonitoring();
      });

      qualityBtn?.addEventListener('click', () => {
        this.toggleQuality();
      });

      speedBtn?.addEventListener('click', () => {
        this.toggleSpeedMode();
      });
    },

    toggleMonitoring() {
      this.isEnabled = !this.isEnabled;
      const toggleBtn = this.overlay?.querySelector('.toggle-btn');
      
      if (this.isEnabled) {
        this.startVideoMonitoring();
        if (toggleBtn) toggleBtn.textContent = '⏸️';
        this.updateOverlay({ status: "Monitoring resumed", is_deepfake: false, confidence: 0 });
      } else {
        this.stopMonitoring();
        if (toggleBtn) toggleBtn.textContent = '▶️';
        this.updateOverlay({ status: "Monitoring paused", is_deepfake: false, confidence: 0 });
      }
    },

    stopMonitoring() {
      if (this.analysisInterval) {
        clearInterval(this.analysisInterval);
        this.analysisInterval = null;
      }
      this.isAnalyzing = false;
    },

    toggleQuality() {
      const qualities = ['low', 'medium', 'high'];
      const currentIndex = qualities.indexOf(this.frameQuality);
      this.frameQuality = qualities[(currentIndex + 1) % qualities.length];
      
      const qualityBtn = this.overlay?.querySelector('.quality-btn');
      const labels = { low: 'LQ', medium: 'MD', high: 'HQ' };
      if (qualityBtn) {
        qualityBtn.textContent = labels[this.frameQuality];
      }
    },

    toggleSpeedMode() {
      const speedBtn = this.overlay?.querySelector('.speed-btn');
      
      if (this.config.analysisIntervalMs > 2000) {
        // Fast mode
        this.config.analysisIntervalMs = 1500;
        this.config.minTimeBetweenAnalysis = 1200;
        speedBtn.textContent = '⚡';
        speedBtn.classList.add('active');
      } else {
        // Normal mode
        this.config.analysisIntervalMs = 3000;
        this.config.minTimeBetweenAnalysis = 2500;
        speedBtn.textContent = '🚀';
        speedBtn.classList.remove('active');
      }
      
      if (this.analysisInterval) {
        this.startVideoMonitoring(); // Restart with new interval
      }
    },

    startVideoMonitoring() {
      if (!this.isEnabled) return;
      
      // Clear any existing interval
      if (this.analysisInterval) {
        clearInterval(this.analysisInterval);
      }

      this.analysisInterval = setInterval(() => {
        this.checkForVideoAndAnalyze();
      }, this.config.analysisIntervalMs);

      // Run immediately
      setTimeout(() => this.checkForVideoAndAnalyze(), 100);
    },

    checkForVideoAndAnalyze() {
      if (!this.isEnabled || this.isAnalyzing) return;
      
      const now = Date.now();
      
      // Throttle analysis
      if (now - this.lastAnalysisTime < this.config.minTimeBetweenAnalysis) {
        return;
      }

      const videoElement = this.findBestVideo();
      
      if (!videoElement) {
        this.updateOverlay({ 
          status: "No video found", 
          is_deepfake: false, 
          confidence: 0 
        });
        return;
      }

      // Reset history if video changed
      if (videoElement !== this.currentVideo) {
        this.currentVideo = videoElement;
        this.analysisHistory = [];
        this.lastFrameHash = null;
        console.log('📹 New video detected');
      }

      this.analyzeVideo(videoElement);
      this.lastAnalysisTime = now;
    },

    findBestVideo() {
      // Platform-specific video detection
      const currentHost = window.location.hostname;
      
      // Google Meet specific selectors
      if (currentHost.includes('meet.google.com')) {
        const meetSelectors = [
          'video[autoplay]',
          'video[src*="blob:"]',
          'video[srcObject]',
          '[data-allocation-index] video',
          '.u6vdEc video', // Meet participant video container
          '[jsname] video',
          'video'
        ];
        
        for (const selector of meetSelectors) {
          const videos = Array.from(document.querySelectorAll(selector));
          for (const video of videos) {
            if (this.isVideoSuitableForAnalysis(video) && this.isVisibleVideo(video)) {
              console.log('Found Google Meet video:', video);
              return video;
            }
          }
        }
      }
      
      // Zoom specific selectors
      else if (currentHost.includes('zoom.us')) {
        const zoomSelectors = [
          'video[autoplay]',
          'video[id*="video"]',
          '.video-container video',
          '[id*="participant"] video',
          '[class*="video"] video',
          'video[src*="blob:"]',
          'video[srcObject]',
          'video'
        ];
        
        for (const selector of zoomSelectors) {
          const videos = Array.from(document.querySelectorAll(selector));
          for (const video of videos) {
            if (this.isVideoSuitableForAnalysis(video) && this.isVisibleVideo(video)) {
              console.log('Found Zoom video:', video);
              return video;
            }
          }
        }
      }
      
      // Microsoft Teams specific selectors
      else if (currentHost.includes('teams.microsoft.com')) {
        const teamsSelectors = [
          'video[autoplay]',
          '.fui-Flex video',
          '[data-tid*="video"] video',
          '.video-stream video',
          'video[src*="blob:"]',
          'video[srcObject]',
          'video'
        ];
        
        for (const selector of teamsSelectors) {
          const videos = Array.from(document.querySelectorAll(selector));
          for (const video of videos) {
            if (this.isVideoSuitableForAnalysis(video) && this.isVisibleVideo(video)) {
              console.log('Found Teams video:', video);
              return video;
            }
          }
        }
      }
      
      // Webex specific selectors
      else if (currentHost.includes('webex.com')) {
        const webexSelectors = [
          'video[autoplay]',
          '.video-container video',
          '[data-testid*="video"] video',
          '.participant-video video',
          'video[src*="blob:"]',
          'video[srcObject]',
          'video'
        ];
        
        for (const selector of webexSelectors) {
          const videos = Array.from(document.querySelectorAll(selector));
          for (const video of videos) {
            if (this.isVideoSuitableForAnalysis(video) && this.isVisibleVideo(video)) {
              console.log('Found Webex video:', video);
              return video;
            }
          }
        }
      }
      
      // Discord specific selectors
      else if (currentHost.includes('discord.com')) {
        const discordSelectors = [
          'video[autoplay]',
          '.video-layer video',
          '[class*="video"] video',
          '.participant-video video',
          'video[src*="blob:"]',
          'video[srcObject]',
          'video'
        ];
        
        for (const selector of discordSelectors) {
          const videos = Array.from(document.querySelectorAll(selector));
          for (const video of videos) {
            if (this.isVideoSuitableForAnalysis(video) && this.isVisibleVideo(video)) {
              console.log('Found Discord video:', video);
              return video;
            }
          }
        }
      }
      
      // TikTok specific selectors
      else if (currentHost.includes('tiktok.com')) {
        const tiktokSelectors = [
          'video[autoplay]',
          '[data-e2e="video-player"] video',
          '.video-player video',
          '.xgplayer-video video',
          'video[playsinline]',
          'video'
        ];
        
        for (const selector of tiktokSelectors) {
          const videos = Array.from(document.querySelectorAll(selector));
          for (const video of videos) {
            if (this.isVideoSuitableForAnalysis(video) && this.isVisibleVideo(video)) {
              console.log('Found TikTok video:', video);
              return video;
            }
          }
        }
      }
      
      // Facebook/Instagram specific selectors
      else if (currentHost.includes('facebook.com') || currentHost.includes('instagram.com')) {
        const fbSelectors = [
          'video[autoplay]',
          '[role="main"] video',
          '[data-testid*="video"] video',
          '.video-player video',
          'video[playsinline]',
          'video[src*="blob:"]',
          'video[srcObject]',
          'video'
        ];
        
        for (const selector of fbSelectors) {
          const videos = Array.from(document.querySelectorAll(selector));
          for (const video of videos) {
            if (this.isVideoSuitableForAnalysis(video) && this.isVisibleVideo(video)) {
              console.log('Found Facebook/Instagram video:', video);
              return video;
            }
          }
        }
      }
      
      // Twitch specific selectors
      else if (currentHost.includes('twitch.tv')) {
        const twitchSelectors = [
          'video[autoplay]',
          '.video-player video',
          '[data-a-target="video-player"] video',
          'video[src*="blob:"]',
          'video'
        ];
        
        for (const selector of twitchSelectors) {
          const videos = Array.from(document.querySelectorAll(selector));
          for (const video of videos) {
            if (this.isVideoSuitableForAnalysis(video) && this.isVisibleVideo(video)) {
              console.log('Found Twitch video:', video);
              return video;
            }
          }
        }
      }
      
      // Skype Web specific selectors
      else if (currentHost.includes('skype.com')) {
        const skypeSelectors = [
          'video[autoplay]',
          '.video-stream video',
          '[data-tid*="video"] video',
          'video[src*="blob:"]',
          'video[srcObject]',
          'video'
        ];
        
        for (const selector of skypeSelectors) {
          const videos = Array.from(document.querySelectorAll(selector));
          for (const video of videos) {
            if (this.isVideoSuitableForAnalysis(video) && this.isVisibleVideo(video)) {
              console.log('Found Skype video:', video);
              return video;
            }
          }
        }
      }
      
      // General video detection (fallback)
      else {
        const generalSelectors = [
          'video[autoplay]:not([muted])',
          'video:not([muted])',
          'video[autoplay]',
          'video'
        ];

        for (const selector of generalSelectors) {
          const videos = Array.from(document.querySelectorAll(selector));
          videos.sort((a, b) => (b.offsetWidth * b.offsetHeight) - (a.offsetWidth * a.offsetHeight));
          
          for (const video of videos) {
            if (this.isVideoSuitableForAnalysis(video) && this.isVisibleVideo(video)) {
              return video;
            }
          }
        }
      }

      return null;
    },

    isVisibleVideo(video) {
      // Check if video is actually visible on screen
      const rect = video.getBoundingClientRect();
      const style = window.getComputedStyle(video);
      
      return (
        rect.width > 50 &&
        rect.height > 50 &&
        rect.top < window.innerHeight &&
        rect.bottom > 0 &&
        rect.left < window.innerWidth &&
        rect.right > 0 &&
        style.display !== 'none' &&
        style.visibility !== 'hidden' &&
        parseFloat(style.opacity) > 0.1
      );
    },

    isVideoSuitableForAnalysis(video) {
      // More lenient criteria for video call platforms
      const isVideoCall = window.location.hostname.includes('meet.google.com') || 
                         window.location.hostname.includes('zoom.us') ||
                         window.location.hostname.includes('teams.microsoft.com');
      
      if (isVideoCall) {
        // More relaxed criteria for video calls
        return (
          video.readyState >= 1 && // HAVE_METADATA is enough for video calls
          video.videoWidth > 0 && 
          video.videoHeight > 0 &&
          video.offsetWidth > 50 && // Smaller minimum size for video calls
          video.offsetHeight > 50 &&
          // Don't require playing state for video calls (they might be paused)
          (video.currentTime > 0 || video.readyState >= 2)
        );
      } else {
        // Original criteria for streaming platforms
        return (
          video.readyState >= 2 &&
          video.videoWidth > 0 && 
          video.videoHeight > 0 &&
          !video.paused &&
          video.currentTime > 0 &&
          video.offsetWidth > 80 &&
          video.offsetHeight > 80 &&
          !video.muted
        );
      }
    },

    analyzeVideo(videoElement) {
      if (this.isAnalyzing || !this.isEnabled) {
        return;
      }

      this.isAnalyzing = true;
      const startTime = Date.now();
      
      this.updateOverlay({ 
        status: "Analyzing...", 
        is_deepfake: false, 
        confidence: 0 
      });

      try {
        const frame = this.captureOptimizedFrame(videoElement);
        if (frame) {
          // Check if frame is identical to last one
          const frameHash = this.getSimpleFrameHash(frame);
          if (this.config.skipIdenticalFrames && frameHash === this.lastFrameHash) {
            console.log('Skipping identical frame');
            this.isAnalyzing = false;
            return;
          }
          this.lastFrameHash = frameHash;
          
          this.sendFrameForAnalysis(frame, startTime);
        } else {
          this.isAnalyzing = false;
          this.updateOverlay({ 
            status: "Capture failed", 
            is_deepfake: false, 
            confidence: 0 
          });
        }
      } catch (error) {
        console.error('Video analysis error:', error);
        this.isAnalyzing = false;
        this.updateOverlay({ 
          status: "Analysis error", 
          is_deepfake: false, 
          confidence: 0 
        });
      }
    },

    getSimpleFrameHash(imageDataUrl) {
      // Simple hash based on data length and first few characters
      return imageDataUrl.length + imageDataUrl.substring(50, 100);
    },

    captureOptimizedFrame(videoElement) {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d', { alpha: false }); // Disable alpha for performance

      const videoWidth = videoElement.videoWidth;
      const videoHeight = videoElement.videoHeight;
      const aspectRatio = videoWidth / videoHeight;

      // Optimized dimensions based on quality setting
      let canvasWidth, canvasHeight;
      switch (this.frameQuality) {
        case 'low':
          canvasWidth = Math.min(videoWidth, 224);
          break;
        case 'medium':
          canvasWidth = Math.min(videoWidth, 384);
          break;
        case 'high':
          canvasWidth = Math.min(videoWidth, 512);
          break;
      }
      
      canvasHeight = canvasWidth / aspectRatio;
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;

      try {
        // Optimized drawing settings
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = this.frameQuality === 'high' ? 'high' : 'medium';
        ctx.drawImage(videoElement, 0, 0, canvasWidth, canvasHeight);

        // Adjust quality based on performance needs
        const quality = this.frameQuality === 'high' ? 0.8 : 0.75;
        return canvas.toDataURL('image/jpeg', quality);
      } catch (error) {
        console.error('Frame capture error:', error);
        return null;
      }
    },

    async sendFrameForAnalysis(imageDataUrl, startTime) {
      const requestStartTime = Date.now();
      
      try {
        const response = await fetch('http://127.0.0.1:5000/analyze', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ frame: imageDataUrl }),
        });

        const result = await response.json();
        const processingTime = Date.now() - requestStartTime;
        
        this.isAnalyzing = false;

        if (response.ok && result.confidence >= this.config.confidenceThreshold) {
          this.processAnalysisResult(result, processingTime);
        } else {
          console.warn('Low confidence result or server error:', result);
          this.updateOverlay({ 
            status: result.error || "Uncertain result", 
            is_deepfake: false, 
            confidence: result.confidence || 0,
            processingTime
          });
        }
      } catch (error) {
        console.error('Network error:', error);
        this.isAnalyzing = false;
        
        // Show different messages based on error type
        let errorMsg = "Connection failed";
        if (error.name === 'TypeError') {
          errorMsg = "Server offline";
        } else if (error.message.includes('timeout')) {
          errorMsg = "Analysis timeout";
        }
        
        this.updateOverlay({ 
          status: errorMsg, 
          is_deepfake: false, 
          confidence: 0,
          processingTime: Date.now() - requestStartTime
        });
      }
    },

    processAnalysisResult(result, processingTime) {
      const isDeepfake = result.prediction === "Deepfake";
      
      // Add to history with reduced size
      this.analysisHistory.push({
        timestamp: Date.now(),
        isDeepfake,
        confidence: result.confidence,
        score: result.score,
        processingTime
      });

      // Keep history size small for performance
      if (this.analysisHistory.length > this.config.historySize) {
        this.analysisHistory.shift();
      }

      // Quick consecutive detection check
      const recentHistory = this.analysisHistory.slice(-this.config.consecutiveDetectionThreshold);
      const consecutiveDeepfakes = recentHistory.filter(r => r.isDeepfake).length;
      
      // Determine status with faster response
      let finalStatus, showAlert = false;
      if (isDeepfake && consecutiveDeepfakes >= this.config.consecutiveDetectionThreshold) {
        finalStatus = "⚠️ DEEPFAKE DETECTED";
        showAlert = true;
      } else if (isDeepfake) {
        finalStatus = "Potential deepfake";
      } else if (result.confidence > 0.8) {
        finalStatus = "Video authentic";
      } else if (result.confidence > 0.6) {
        finalStatus = "Likely authentic";
      } else {
        finalStatus = "Analysis uncertain";
      }

      // Show notification for high-confidence deepfakes
      if (showAlert && result.confidence > 0.75) {
        this.showDeepfakeAlert(result);
      }

      // Update display
      this.updateOverlay({
        status: finalStatus,
        is_deepfake: isDeepfake && consecutiveDeepfakes >= this.config.consecutiveDetectionThreshold,
        confidence: result.confidence,
        score: result.score,
        processingTime,
        details: result.details
      });

      // Auto-clear status after shorter duration
      setTimeout(() => {
        if (!this.isAnalyzing) {
          this.updateOverlay({
            status: "Monitoring...",
            is_deepfake: false,
            confidence: 0,
            processingTime: 0
          });
        }
      }, this.config.statusDisplayDuration);

      // Adaptive interval based on results
      if (this.config.adaptiveInterval) {
        if (isDeepfake && result.confidence > 0.8) {
          // Analyze more frequently if deepfake detected
          this.config.analysisIntervalMs = Math.max(1500, this.config.analysisIntervalMs * 0.8);
        } else if (!isDeepfake && result.confidence > 0.8) {
          // Analyze less frequently if confident it's real
          this.config.analysisIntervalMs = Math.min(4000, this.config.analysisIntervalMs * 1.1);
        }
      }
    },

    showDeepfakeAlert(result) {
      if ("Notification" in window && Notification.permission === "granted") {
        new Notification("🚨 DeepFake Detected!", {
          body: `Confidence: ${Math.round(result.confidence * 100)}%`,
          requireInteraction: true,
          tag: 'deepfake-alert' // Prevent duplicate notifications
        });
      }
    },

    updateOverlay(data) {
      if (!this.overlay) return;

      const statusText = this.overlay.querySelector('.status-text');
      const confidenceBar = this.overlay.querySelector('.confidence-fill');
      const scoreSpan = this.overlay.querySelector('.score');
      const timingSpan = this.overlay.querySelector('.timing');

      if (statusText) {
        statusText.textContent = data.status;
      }

      if (confidenceBar && typeof data.confidence === 'number') {
        confidenceBar.style.width = `${Math.min(data.confidence * 100, 100)}%`;
      }

      if (scoreSpan && typeof data.score === 'number') {
        scoreSpan.textContent = `Score: ${data.score.toFixed(2)}`;
      }

      if (timingSpan && data.processingTime) {
        timingSpan.textContent = `Time: ${data.processingTime}ms`;
      }

      // Update overlay appearance with performance optimizations
      this.overlay.className = '';
      
      if (data.is_deepfake) {
        this.overlay.classList.add('deepfake-detected');
        if (data.confidence > 0.8) {
          this.overlay.classList.add('pulse');
        }
      } else if (data.status === "Analyzing...") {
        this.overlay.classList.add('analyzing');
      } else if (data.confidence > this.config.confidenceThreshold) {
        this.overlay.classList.add('real-detected');
      }

      // Remove pulse animation after completion
      setTimeout(() => {
        this.overlay.classList.remove('pulse');
      }, 1000);
    },

    setupPerformanceMonitoring() {
      // Monitor performance and adjust settings automatically
      let analysisCount = 0;
      let totalProcessingTime = 0;
      
      setInterval(() => {
        if (analysisCount > 0) {
          const avgProcessingTime = totalProcessingTime / analysisCount;
          
          // Auto-adjust quality based on performance
          if (avgProcessingTime > 3000 && this.frameQuality === 'high') {
            console.log('Performance: Reducing quality due to slow processing');
            this.frameQuality = 'medium';
            this.toggleQuality(); // Update UI
          } else if (avgProcessingTime < 1000 && this.frameQuality === 'low') {
            console.log('Performance: Increasing quality due to fast processing');
            this.frameQuality = 'medium';
            this.toggleQuality(); // Update UI
          }
          
          // Reset counters
          analysisCount = 0;
          totalProcessingTime = 0;
        }
      }, 30000); // Check every 30 seconds
    },

    addKeyboardShortcuts() {
      document.addEventListener('keydown', (e) => {
        // Ctrl+Shift+D: Toggle monitoring
        if (e.ctrlKey && e.shiftKey && e.key === 'D') {
          e.preventDefault();
          this.toggleMonitoring();
        }
        
        // Ctrl+Shift+Q: Toggle quality
        if (e.ctrlKey && e.shiftKey && e.key === 'Q') {
          e.preventDefault();
          this.toggleQuality();
        }
        
        // Ctrl+Shift+S: Toggle speed mode
        if (e.ctrlKey && e.shiftKey && e.key === 'S') {
          e.preventDefault();
          this.toggleSpeedMode();
        }
      });
    },

    addMessageListener() {
      chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        switch (request.action) {
          case "updateOverlay":
            this.updateOverlay(request.result);
            break;
          case "toggleMonitoring":
            this.toggleMonitoring();
            break;
          case "ping":
            sendResponse({ status: "active" });
            break;
          case "settingChanged":
            this.handleSettingChange(request.setting, request.value);
            break;
        }
      });
    },

    handleSettingChange(setting, value) {
      switch (setting) {
        case 'autoMonitor':
          this.isEnabled = value;
          if (value) {
            this.startVideoMonitoring();
          } else {
            this.stopMonitoring();
          }
          break;
        case 'highQuality':
          this.frameQuality = value ? 'high' : 'medium';
          break;
      }
    }
  };

  // Initialize the content script
  contentScript.initialize();
})();

















