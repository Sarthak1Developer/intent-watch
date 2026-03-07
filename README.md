# IntentWatch AI Surveillance System

A real-time AI-powered CCTV surveillance system with intent detection capabilities.

## Features

- **Real-time Video Analysis**: Process live webcam feeds or uploaded videos
- **AI-Powered Detection**: YOLOv8-based object detection
- **Intent Recognition**: Detect behaviors like loitering, running, and unattended bags
- **Live Alerts**: Real-time alert system with comprehensive logging
- **Modern UI**: React-based dashboard with real-time updates
- **Analytics**: View detection statistics and alert history

## Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **YOLOv8**: State-of-the-art object detection
- **OpenCV**: Video processing and computer vision
- **Ultralytics**: YOLO model implementation

### Frontend
- **React**: Modern UI library
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **Recharts**: Data visualization

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn
- Webcam (optional, for live feed)

## Installation

### 1. Clone the Repository

```powershell
cd IntentWatch
```

### 2. Set Up Python Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```powershell
cd "Build AI Surveillance System"
npm install
cd ..
```

## Running the Application

### Option 1: Start Everything (Recommended)

Run both backend and frontend with a single command:

```powershell
.\start-intentwatch.ps1
```

This will open two terminals:
- Backend server on http://localhost:8000
- Frontend dev server on http://localhost:5173

### Option 2: Start Manually

#### Start Backend:
```powershell
.\start-backend.ps1
```

#### Start Frontend (in a new terminal):
```powershell
.\start-frontend.ps1
```

### Option 3: Run Components Separately

#### Backend:
```powershell
.\venv\Scripts\Activate.ps1
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend:
```powershell
cd "Build AI Surveillance System"
npm run dev
```

## Usage

1. **Access the Application**
   - Open your browser to http://localhost:5173
   - The backend API will be available at http://localhost:8000

2. **Start Detection**
   - Navigate to "Live Feed" page
   - Choose a video source (Webcam or upload a video file)
   - Click "Start" to begin detection

3. **View Alerts**
   - Real-time alerts appear in the Live Feed sidebar
   - Go to "Alerts Log" to see complete alert history
   - Use filters to search and sort alerts

4. **Monitor Analytics**
   - Dashboard shows real-time statistics
   - Analytics page provides detailed graphs and insights

## Project Structure

```
IntentWatch/
├── backend/                    # FastAPI backend
│   ├── api/
│   │   ├── main.py            # FastAPI app entry point
│   │   └── routes/            # API endpoints
│   │       ├── video.py       # Video processing routes
│   │       └── alerts.py      # Alert management routes
│   ├── data/videos/           # Uploaded videos storage
│   └── requirements.txt       # Python dependencies
│
├── Build AI Surveillance System/  # React frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── App.tsx        # Main app component
│   │   │   ├── pages/         # Application pages
│   │   │   └── components/    # Reusable components
│   │   └── services/
│   │       └── api.ts         # Backend API integration
│   ├── package.json
│   └── vite.config.ts         # Vite configuration
│
├── start-intentwatch.ps1      # Start both servers
├── start-backend.ps1          # Start backend only
├── start-frontend.ps1         # Start frontend only
└── README.md                  # This file
```

## API Endpoints

### Video Processing
- `POST /video/upload` - Upload a video file
- `POST /video/start` - Start video processing
- `POST /video/stop` - Stop video processing
- `POST /video/start-camera` - Start webcam stream
- `GET /video/stream` - Get video stream
- `GET /video/status` - Get stream status

### Alerts
- `GET /alerts/live` - Get all alerts
- `GET /alerts/analytics` - Get analytics data
- `POST /alerts/clear` - Clear all alerts

## Configuration

### Backend Configuration
Edit `backend/api/routes/video.py` to adjust:
- `LOITER_THRESHOLD`: Time in seconds for loitering detection
- `BAG_THRESHOLD`: Time in seconds for unattended bag detection
- `RUNNING_SPEED_THRESHOLD`: Speed threshold for running detection

### Frontend Configuration
Edit `.env` file in the frontend directory:
```
VITE_API_URL=http://localhost:8000
```

## Troubleshooting

### Backend Issues

**Error: "Module not found"**
```powershell
pip install -r requirements.txt
```

**Error: "YOLO model not found"**
- Ensure `yolov8n.pt` is in the root directory
- It will be auto-downloaded on first run

### Frontend Issues

**Error: "Cannot find module"**
```powershell
cd "Build AI Surveillance System"
npm install
```

**Error: "Port 5173 is already in use"**
- Change the port in `vite.config.ts`
- Or kill the process using port 5173

### Connection Issues

**Frontend can't connect to backend**
- Ensure backend is running on port 8000
- Check CORS settings in `backend/api/main.py`
- Verify `.env` file has correct API URL

## Development

### Adding New Detection Features

1. Modify `backend/api/routes/video.py`
2. Add detection logic in the stream processing loop
3. Use `add_alert()` to trigger alerts
4. Update frontend to display new alert types

### Customizing the UI

- Frontend pages are in `Build AI Surveillance System/src/app/pages/`
- Components are in `Build AI Surveillance System/src/app/components/`
- Styles use Tailwind CSS classes

## Performance Tips

- Use GPU acceleration if available (requires CUDA setup)
- Adjust video resolution for better performance
- Reduce FPS for lower resource usage
- Use video files instead of webcam for testing

## License

This project is for educational and development purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at http://localhost:8000/docs
- Check console logs for errors

## Future Enhancements

- [ ] Multi-camera support
- [ ] Face recognition
- [ ] License plate detection
- [ ] Cloud deployment
- [ ] Mobile app
- [ ] Email/SMS alerts
- [ ] Video playback and review
- [ ] Zone management UI
- [ ] User authentication

---

**Made with ❤️ by [Sarthak Maheshwari](https://github.com/Sarthak1Developer)**

IntentWatch - AI Security Surveillance System
for surveillance and security applications
