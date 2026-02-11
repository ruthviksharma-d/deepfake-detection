// background.js (MV3 service worker, classic script)
// importScripts('vendor/tf.min.js'); // loads self.tf

// const tf = self.tf; // convenience

// class DeepFakeDetector {
//   constructor() {
//     this.artifactModel = null;
//     this.syncModel = null;
//     this.isInitialized = false;
//     this.frameBuffer = [];
//     this.analysisWindow = 3000; // 3 seconds
//     this.lastScores = [];
//   }

//   async initialize() {
//     if (this.isInitialized) return true;
//     try {
//       // Try to load packaged TFJS models if present (optional)
//       // If not present, fallback path will be used
//       const artifactUrl = chrome.runtime.getURL('models/efficientnet_lite/model.json');
//       const syncUrl     = chrome.runtime.getURL('models/syncnet_lite/model.json');

//       // Attempt artifact model
//       try {
//         this.artifactModel = await tf.loadLayersModel(artifactUrl);
//       } catch (e) {
//         console.warn('Artifact model not found, using fallback:', e.message);
//         this.artifactModel = null;
//       }

//       // Attempt sync model
//       try {
//         this.syncModel = await tf.loadLayersModel(syncUrl);
//       } catch (e) {
//         console.warn('Sync model not found, using fallback:', e.message);
//         this.syncModel = null;
//       }

//       this.isInitialized = true;
//       return true;
//     } catch (err) {
//       console.error('Model init failed, fallback only:', err);
//       this.isInitialized = true; // still allow fallback
//       return true;
//     }
//   }

//   preprocessFrame(buffer) {
//     // buffer is an ArrayBuffer of Uint8ClampedArray (RGBA)
//     const w = 224, h = 224, ch = 4;
//     const arr = new Uint8ClampedArray(buffer);
//     const rgba = tf.tensor4d(arr, [1, h, w, ch]);
//     const rgb  = rgba.slice([0,0,0,0], [1, h, w, 3]).toFloat().div(255);
//     rgba.dispose();
//     return rgb; // [1,224,224,3]
//   }

//   async detectArtifacts(frameTensor) {
//     if (!this.artifactModel) return this.fallbackArtifactDetection(frameTensor);
//     try {
//       const pred = this.artifactModel.predict(frameTensor);
//       const data = await pred.data();
//       pred.dispose();
//       const s = data[0]; // assume single logit/sigmoid
//       return { artifactScore: s, confidence: Math.min(Math.abs(s - 0.5) * 2, 1) };
//     } catch (e) {
//       console.warn('Artifact model error, fallback:', e.message);
//       return this.fallbackArtifactDetection(frameTensor);
//     }
//   }

//   fallbackArtifactDetection(frameTensor) {
//     // Very rough heuristic on pixel variance / “texture weirdness”
//     const flat = frameTensor.reshape([-1]);
//     const mean = flat.mean();
//     const variance = flat.sub(mean).square().mean();
//     const v = variance.dataSync()[0];
//     flat.dispose(); mean.dispose(); variance.dispose();
//     const artifactScore = Math.max(0, Math.min(v * 10, 1));
//     return { artifactScore, confidence: 0.3 };
//   }

//   async analyzeTemporalConsistency() {
//     if (this.frameBuffer.length < 3) return { consistency: 0.8, score: 0.1 };
//     let total = 0, n = 0;
//     for (let i = 1; i < this.frameBuffer.length; i++) {
//       const curr = this.frameBuffer[i].tensor;
//       const prev = this.frameBuffer[i-1].tensor;
//       const diff = curr.sub(prev).abs().mean();
//       total += diff.dataSync()[0];
//       n++;
//       diff.dispose();
//     }
//     const avgDiff = total / n;
//     const incScore = Math.min(avgDiff * 5, 1);
//     return { consistency: 1 - incScore, score: incScore };
//   }

//   async basicLipSyncCheck(frameTensor) {
//     // Placeholder “mouth activity” proxy
//     const lowerFace = frameTensor.slice([0, 150, 75, 0], [1, 74, 74, 3]);
//     const mv = lowerFace.sub(lowerFace.mean()).square().mean().dataSync()[0];
//     lowerFace.dispose();
//     const syncScore = Math.min(mv * 2, 1);
//     return { syncScore, confidence: 0.4 };
//   }

//   async analyzeFrame(buffer, timestamp) {
//     const frameTensor = this.preprocessFrame(buffer);
//     // buffer for temporal analysis
//     this.frameBuffer.push({ tensor: frameTensor.clone(), timestamp });
//     const cutoff = timestamp - this.analysisWindow;
//     const keep = [];
//     for (const f of this.frameBuffer) {
//       if (f.timestamp > cutoff) keep.push(f); else f.tensor.dispose();
//     }
//     this.frameBuffer = keep;

//     try {
//       const [artifact, temporal, sync] = await Promise.all([
//         this.detectArtifacts(frameTensor),
//         this.analyzeTemporalConsistency(),
//         this.basicLipSyncCheck(frameTensor)
//       ]);

//       const score =
//         artifact.artifactScore * 0.5 +
//         temporal.score          * 0.3 +
//         sync.syncScore          * 0.2;

//       this.lastScores.push(score);
//       if (this.lastScores.length > 5) this.lastScores.shift();
//       const smoothed = this.lastScores.reduce((a,b)=>a+b,0) / this.lastScores.length;

//       frameTensor.dispose();

//       const confidence = Math.min(
//         (artifact.confidence + temporal.consistency + sync.confidence) / 3, 1
//       );

//       return {
//         score: smoothed,
//         confidence,
//         temporalConsistency: temporal.consistency,
//         breakdown: {
//           artifact: artifact.artifactScore,
//           temporal: temporal.score,
//           sync: sync.syncScore
//         }
//       };
//     } catch (e) {
//       console.error('Analyze error:', e);
//       frameTensor.dispose();
//       return { score: 0.5, confidence: 0.1, temporalConsistency: 0.5, error: e.message };
//     }
//   }

//   cleanup() {
//     for (const f of this.frameBuffer) f.tensor.dispose();
//     this.frameBuffer = [];
//     this.lastScores = [];
//   }
// }

// let detector = null;

// chrome.runtime.onMessage.addListener((req, sender, sendResponse) => {
//   if (req.action === 'initializeML') {
//     (async () => {
//       if (!detector) detector = new DeepFakeDetector();
//       const ok = await detector.initialize();
//       sendResponse({ success: ok, initialized: detector.isInitialized });
//     })();
//     return true;
//   }

//   if (req.action === 'analyzeFrame') {
//     (async () => {
//       if (!detector || !detector.isInitialized) {
//         sendResponse({ error: 'Detector not initialized' });
//         return;
//       }
//       const result = await detector.analyzeFrame(req.buffer, req.timestamp);
//       sendResponse(result);
//     })();
//     return true;
//   }

//   if (req.action === 'cleanup') {
//     if (detector) detector.cleanup();
//     sendResponse({ success: true });
//   }
// });



















// Enhanced background.js (MV3 service worker) ~85% accuracy

class DeepFakeShieldBackground {
  constructor() {
    this.activeAnalyses = new Map();
    this.serverStatus = 'unknown';
    this.lastServerCheck = 0;
    this.serverCheckInterval = 30000; // Check server every 30 seconds
    this.statistics = {
      totalAnalyses: 0,
      deepfakesDetected: 0,
      averageConfidence: 0,
      lastResetTime: Date.now()
    };
    
    this.initialize();
  }

  async initialize() {
    console.log('🛡️ DeepFake Shield Background - Starting...');
    
    // Load saved statistics
    await this.loadStatistics();
    
    // Check server status on startup
    await this.checkServerStatus();
    
    // Set up periodic server checks
    this.setupPeriodicChecks();
    
    // Set up command handlers
    this.setupCommandHandlers();
    
    console.log('✅ Background service initialized');
  }

  async loadStatistics() {
    try {
      const result = await chrome.storage.local.get(['statistics']);
      if (result.statistics) {
        this.statistics = { ...this.statistics, ...result.statistics };
      }
    } catch (error) {
      console.error('Error loading statistics:', error);
    }
  }

  async saveStatistics() {
    try {
      await chrome.storage.local.set({ statistics: this.statistics });
    } catch (error) {
      console.error('Error saving statistics:', error);
    }
  }

  async checkServerStatus() {
    const now = Date.now();
    
    // Don't check too frequently
    if (now - this.lastServerCheck < 5000) {
      return this.serverStatus;
    }
    
    try {
      const response = await fetch('http://127.0.0.1:5000/health', {
        method: 'GET',
        timeout: 5000
      });
      
      if (response.ok) {
        const data = await response.json();
        this.serverStatus = 'online';
        this.lastServerCheck = now;
        
        // Update badge to show server is online
        this.updateBadge('online');
        
        console.log('✅ Server status: Online -', data);
        return 'online';
      } else {
        throw new Error(`Server responded with status ${response.status}`);
      }
    } catch (error) {
      console.warn('⚠️ Server check failed:', error.message);
      this.serverStatus = 'offline';
      this.lastServerCheck = now;
      
      // Update badge to show server is offline
      this.updateBadge('offline');
      
      return 'offline';
    }
  }

  updateBadge(status) {
    const badgeConfig = {
      online: { text: '✓', color: '#2ed573' },
      offline: { text: '✗', color: '#ff4757' },
      analyzing: { text: '•', color: '#ffa502' }
    };

    const config = badgeConfig[status] || badgeConfig.offline;
    
    chrome.action.setBadgeText({ text: config.text });
    chrome.action.setBadgeBackgroundColor({ color: config.color });
  }

  setupPeriodicChecks() {
    // Check server status periodically
    setInterval(() => {
      this.checkServerStatus();
    }, this.serverCheckInterval);
  }

  setupCommandHandlers() {
    // Handle keyboard shortcuts
    chrome.commands.onCommand.addListener((command) => {
      if (command === 'toggle-monitoring') {
        this.broadcastToContentScripts({ action: 'toggleMonitoring' });
      }
    });
  }

  async broadcastToContentScripts(message) {
    try {
      const tabs = await chrome.tabs.query({});
      
      for (const tab of tabs) {
        // Only send to tabs that might have our content script
        if (this.isVideoSite(tab.url)) {
          try {
            await chrome.tabs.sendMessage(tab.id, message);
          } catch (error) {
            // Ignore errors for tabs without content script
          }
        }
      }
    } catch (error) {
      console.error('Error broadcasting to content scripts:', error);
    }
  }

  isVideoSite(url) {
    if (!url) return false;
    
    const videoSites = [
      'zoom.us', 'meet.google.com', 'teams.microsoft.com',
      'webex.com', 'youtube.com', 'twitch.tv', 'facebook.com',
      'instagram.com', 'tiktok.com', 'discord.com', 'skype.com'
    ];
    
    return videoSites.some(site => url.includes(site));
  }

  updateStatistics(result) {
    this.statistics.totalAnalyses++;
    
    if (result.prediction === 'Deepfake' && result.confidence > 0.6) {
      this.statistics.deepfakesDetected++;
    }
    
    // Update rolling average confidence
    const currentAvg = this.statistics.averageConfidence;
    const newConfidence = result.confidence || 0;
    this.statistics.averageConfidence = 
      (currentAvg * (this.statistics.totalAnalyses - 1) + newConfidence) / this.statistics.totalAnalyses;
    
    // Save statistics periodically
    if (this.statistics.totalAnalyses % 10 === 0) {
      this.saveStatistics();
    }
  }

  async handleAnalysisRequest(request, sender) {
    const tabId = sender.tab?.id;
    if (!tabId) return { error: 'No tab ID' };

    // Check server status first
    const serverStatus = await this.checkServerStatus();
    if (serverStatus !== 'online') {
      return { 
        error: 'Analysis server is offline',
        serverStatus,
        suggestion: 'Please ensure the Flask server is running on localhost:5000'
      };
    }

    // Track active analysis
    this.activeAnalyses.set(tabId, {
      startTime: Date.now(),
      url: sender.tab?.url
    });

    this.updateBadge('analyzing');

    try {
      // Forward request to analysis server
      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(request.data)
      });

      const result = await response.json();
      
      // Update statistics
      this.updateStatistics(result);
      
      // Clean up active analysis tracking
      this.activeAnalyses.delete(tabId);
      this.updateBadge('online');

      return result;

    } catch (error) {
      console.error('Analysis request failed:', error);
      this.activeAnalyses.delete(tabId);
      this.updateBadge('offline');
      
      return {
        error: 'Analysis failed: ' + error.message,
        suggestion: 'Check if the analysis server is running and accessible'
      };
    }
  }

  async handleGetStatistics() {
    return {
      ...this.statistics,
      serverStatus: this.serverStatus,
      activeAnalyses: this.activeAnalyses.size,
      uptime: Date.now() - this.statistics.lastResetTime
    };
  }

  async handleResetStatistics() {
    this.statistics = {
      totalAnalyses: 0,
      deepfakesDetected: 0,
      averageConfidence: 0,
      lastResetTime: Date.now()
    };
    await this.saveStatistics();
    return this.statistics;
  }
}

// Initialize the background service
const backgroundService = new DeepFakeShieldBackground();

// Message handler
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const handleAsync = async () => {
    try {
      switch (request.action) {
        case 'analyzeFrame':
          return await backgroundService.handleAnalysisRequest(request, sender);
        
        case 'getStatistics':
          return await backgroundService.handleGetStatistics();
        
        case 'resetStatistics':
          return await backgroundService.handleResetStatistics();
        
        case 'checkServerStatus':
          return await backgroundService.checkServerStatus();
        
        case 'getServerStatus':
          return { 
            status: backgroundService.serverStatus,
            lastCheck: backgroundService.lastServerCheck
          };
        
        default:
          return { error: 'Unknown action: ' + request.action };
      }
    } catch (error) {
      console.error('Message handler error:', error);
      return { error: error.message };
    }
  };

  // Handle async operations
  handleAsync().then(sendResponse).catch(error => {
    console.error('Async handler error:', error);
    sendResponse({ error: error.message });
  });

  return true; // Keep message channel open for async response
});

// Handle extension installation/update
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('🛡️ DeepFake Shield Pro installed');
    
    // Open options page or show welcome
    chrome.tabs.create({
      url: chrome.runtime.getURL('welcome.html')
    });
  } else if (details.reason === 'update') {
    console.log('🔄 DeepFake Shield Pro updated to version', chrome.runtime.getManifest().version);
  }
});

// Handle tab updates to inject content script if needed
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url && backgroundService.isVideoSite(tab.url)) {
    try {
      // Try to ping content script
      await chrome.tabs.sendMessage(tabId, { action: 'ping' });
    } catch (error) {
      // Content script not loaded, inject it
      try {
        await chrome.scripting.executeScript({
          target: { tabId: tabId },
          files: ['contentScript.js']
        });
        
        await chrome.scripting.insertCSS({
          target: { tabId: tabId },
          files: ['overlay.css']
        });
        
        console.log(`✅ Content script injected into tab ${tabId}`);
      } catch (injectionError) {
        console.warn(`Failed to inject content script into tab ${tabId}:`, injectionError);
      }
    }
  }
});
