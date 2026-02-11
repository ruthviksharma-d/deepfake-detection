# Deepfake Detection

A project for deepfake detection featuring a server backend and a browser extension.

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🚩 Overview

**Deepfake Detection** is an open-source project intended to detect manipulated (“deepfake”) media through machine learning and computer vision techniques. It is designed as a two-part system:
- **Browser Extension:** Interacts with web content and the backend API to analyze media on webpages.
- **Server Backend:** Hosts machine learning models and provides REST API endpoints for deepfake detection.

---

## 📂 Project Structure

```
deepfake-detection/
├── extension/      # Browser extension code
├── server/         # Backend server code (API + ML models)
├── LICENSE         # Project License (MIT)
└── README.md       # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Node.js (v14+ recommended)
- npm
- Modern web browser (Chrome, Firefox)

### 1. Clone the Repository
```sh
git clone https://github.com/ruthviksharma-d/deepfake-detection.git
cd deepfake-detection
```

### 2. Install Dependencies

#### Backend
```sh
cd server
npm install
```

#### Extension
```sh
cd ../extension
npm install
```

### 3. Running the Server

From the `server` directory:

```sh
npm start
```
The backend server will run (by default at `http://localhost:5000`).

#### Environment Variables
Create a `.env` file in `/server` with the following content:
```
PORT=5000
NODE_ENV=development
```

### 4. Loading the Extension

- Go to your browser’s extensions page  
  - Chrome: `chrome://extensions`
  - Firefox: `about:addons`
- Enable "Developer Mode"
- Click "Load unpacked"
- Select the `extension` folder

---

## 🌟 Features

- Real-time browser-based deepfake detection
- REST API for external integrations
- ML-driven video and image analysis
- End-to-end privacy: no user data stored

---

## 📚 API Example

**POST** `/api/detect`
```json
{
  "url": "https://example.com/video.mp4",
  "mediaType": "video"
}
```
**Response**
```json
{
  "isDeepfake": true,
  "confidence": 0.87,
  "details": { ... }
}
```
More endpoints and usage: see `/server` README or code comments.

---

## 🧑‍💻 Contributing

1. Fork this repo
2. Create a feature branch (`git checkout -b my-feature`)
3. Commit your changes
4. Push and open a PR

Issues and suggestions are welcome via [GitHub Issues](https://github.com/ruthviksharma-d/deepfake-detection/issues)!

---

## 📝 License
This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with TensorFlow.js, OpenCV.js and the browser extension APIs
- Inspired by ongoing research in media forensics and misinformation detection

---

**Project by [ruthviksharma-d](https://github.com/ruthviksharma-d)  
Status: Active | Last updated: Feb 2026**
