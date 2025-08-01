# 🌐 VeriFold Web Journal

Beautiful web interface for viewing VeriFold probabilistic observation narratives with real-time updates and GPT-4 integration.

## ✨ Features

### 🎨 **Beautiful UI**
- **Quantum-themed design** with animated backgrounds
- **Emotion-colored glyphs** for each measurement type
- **Responsive layout** that works on desktop and mobile
- **Real-time animations** and smooth transitions

### 📜 **Interactive Timeline**
- **Scrollable journal entries** with rich narratives
- **Live updates** via WebSocket connections
- **Detailed metadata** for each probabilistic observation
- **Verification status** with visual indicators

### 🧠 **AI Integration** 
- **GPT-4 powered summaries** of quantum events
- **Poetic interpretations** of measurement data
- **Symbolic analysis** with emotional context
- **Natural language narratives**

### 📊 **Live Statistics**
- **Real-time metrics** (total events, verification rate, entropy)
- **Emotion tracking** across measurements
- **Performance monitoring**
- **Connection status** indicators

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
cd web_journal/
pip install -r web_journal_requirements.txt
```

### 2. **Set OpenAI API Key** (Optional)
```bash
export OPENAI_API_KEY=your-openai-api-key-here
```

### 3. **Launch Web Journal**
```bash
# Simple launch
python3 launch_web_journal.py

# Or with custom settings
python3 web_journal_app.py --host 0.0.0.0 --port 8080 --debug
```

### 4. **Open in Browser**
Navigate to: http://localhost:5001

## 📁 Project Structure

```
web_journal/
├── web_journal_app.py           # Main Flask application
├── launch_web_journal.py        # Quick launcher script
├── web_journal_requirements.txt # Dependencies
├── templates/
│   └── journal.html             # Main web interface
├── static/                      # Static assets (CSS, JS, images)
└── README.md                    # This file
```

## 🛠️ Configuration

### **Environment Variables**
- `OPENAI_API_KEY`: Your OpenAI API key for GPT-4 summaries
- `VERIFOLD_LOGBOOK`: Path to VeriFold logbook (default: `../verifold_logbook.jsonl`)

### **Command Line Options**
```bash
python3 web_journal_app.py --help

Options:
  --host TEXT     Server host address (default: 0.0.0.0)
  --port INTEGER  Server port (default: 5001)
  --debug         Enable debug mode
```

## 🌟 Interface Overview

### **Main Timeline**
- **Journal Entries**: Each probabilistic observation appears as a narrative card
- **Emotion Glyphs**: Color-coded symbols based on emotional analysis
- **Verification Status**: Green checkmarks for verified, red X for failed
- **Interactive Details**: Click entries to see full metadata

### **Sidebar Panels**

#### 🧠 **AI Synthesis**
- Real-time GPT-4 analysis of recent events
- Poetic summaries with quantum metaphors
- Emotional and symbolic interpretations

#### 📊 **Quantum Stats**
- Total quantum events processed
- Verification success rate
- Average entropy scores
- Active emotion types

#### ⚡ **Live Updates**
- WebSocket connection status
- Real-time event monitoring
- Automatic refresh notifications

## 🎨 Customization

### **Quantum Color Palette**
The interface uses a quantum-inspired color scheme:
- **Primary**: Deep space blue (`#1e3c72`)
- **Secondary**: Quantum blue (`#2a5298`) 
- **Accent**: Bright cyan (`#00d4ff`)
- **Success**: Quantum green (`#4caf50`)
- **Mystery**: Deep purple (`#9c27b0`)

### **Emotion Colors**
Each emotion gets its own color:
- 🔮 **Wonder**: Purple (`#9C27B0`)
- 🔥 **Excitement**: Red (`#F44336`)
- 🧠 **Curiosity**: Blue (`#2196F3`)
- 🎯 **Focus**: Green (`#4CAF50`)
- ❓ **Uncertainty**: Orange (`#FF9800`)
- ✨ **Transcendent**: Pink (`#E91E63`)

## 🔧 Development

### **Adding New Features**
1. **Backend**: Modify `web_journal_app.py` to add new API endpoints
2. **Frontend**: Update `templates/journal.html` for UI changes
3. **Styling**: Add CSS in the `<style>` section of journal.html
4. **JavaScript**: Add functionality in the `<script>` section

### **API Endpoints**
- `GET /`: Main journal interface
- `GET /api/entries`: JSON list of journal entries
- `GET /api/summary`: GPT-4 summary of recent entries
- `GET /api/refresh`: Force refresh of entries

### **WebSocket Events**
- `connect`: Client connection established
- `initial_entries`: Send initial journal entries
- `entries_updated`: Real-time entry updates
- `request_summary`: Client requests GPT summary
- `summary_generated`: GPT summary ready

## 🌐 Deployment

### **Development Server**
```bash
python3 web_journal_app.py --debug
```

### **Production with Gunicorn**
```bash
pip install gunicorn
gunicorn -w 4 --bind 0.0.0.0:5001 web_journal_app:app
```

### **Docker Deployment** (Future)
```dockerfile
# TODO: Add Dockerfile for containerized deployment
```

## 🔍 Troubleshooting

### **Common Issues**

#### **"Flask not available"**
```bash
pip install flask flask-socketio
```

#### **"Journal mode import error"**
- Ensure `journal_mode.py` exists in parent directory
- Check Python path and imports

#### **"GPT summaries not working"**
- Set `OPENAI_API_KEY` environment variable
- Verify OpenAI API key is valid
- Check network connectivity

#### **"No entries showing"**
- Verify `verifold_logbook.jsonl` exists and has valid JSON
- Check file permissions
- Look for errors in browser console

### **Debug Mode**
Run with `--debug` flag for detailed error messages:
```bash
python3 web_journal_app.py --debug
```

## 📱 Mobile Support

The interface is fully responsive and works on:
- **Desktop browsers** (Chrome, Firefox, Safari, Edge)
- **Tablet devices** (iPad, Android tablets)
- **Mobile phones** (iOS Safari, Android Chrome)

### **Mobile Features**
- Touch-friendly navigation
- Responsive grid layouts
- Optimized typography
- Gesture support for scrolling

## 🎯 Future Enhancements

- [ ] **Dark/Light mode toggle**
- [ ] **Custom emotion color themes**
- [ ] **Export journal entries to PDF**
- [ ] **Real-time collaboration features**
- [ ] **Voice narration of entries**
- [ ] **VR/AR visualization modes**
- [ ] **Advanced filtering and search**
- [ ] **User authentication and personalization**

## 🚀 Integration

### **With VeriFold Core**
The web journal automatically connects to:
- `verifold_logbook.jsonl`: Main event log
- `journal_mode.py`: Narrative generation
- `narrative_utils.py`: Emotion and vocabulary systems

### **With External Systems**
- **OpenAI GPT-4**: Poetic summarization
- **WebSocket**: Real-time updates
- **REST API**: Programmatic access

---

**🌟 Experience quantum narratives like never before!**

*Built with ❤️ for the VeriFold Symbolic Intelligence Layer*
