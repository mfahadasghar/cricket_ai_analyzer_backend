# Cricket AI Analyzer - Deployment Guide

## Render Deployment

### Prerequisites
- GitHub account
- Render account (free tier works)
- Git repository with your code

### Step 1: Prepare Repository

1. **Ensure models are committed**:
   ```bash
   git add models/best_stumps.pt models/best_ball.pt
   git commit -m "Add AI models"
   git push
   ```

2. **Verify backend structure**:
   ```
   backend/
   ├── main.py
   ├── requirements.txt
   ├── render.yaml
   ├── .gitignore
   ├── models/
   │   ├── best_stumps.pt
   │   └── best_ball.pt
   ├── detection/
   │   └── stump_detector.py
   └── analysis/
       └── tracker.py
   ```

### Step 2: Deploy to Render

1. **Go to Render Dashboard**: https://dashboard.render.com/

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select your repository

3. **Configure Service**:
   - **Name**: `cricket-ai-analyzer`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`

4. **Environment Variables** (Optional - defaults are set):
   ```
   PORT=8000
   HOST=0.0.0.0
   UPLOAD_DIR=/tmp/uploads
   OUTPUT_DIR=/tmp/outputs
   MODEL_DIR=./models
   ```

5. **Instance Type**:
   - Free tier works but is slow for video processing
   - Recommended: **Starter** ($7/month) or **Standard** ($25/month)
   - Free tier: 512MB RAM, slow startup
   - Starter: 1GB RAM, faster processing

6. **Click "Create Web Service"**

### Step 3: Wait for Deployment

- First deployment takes 5-10 minutes (downloading dependencies + loading models)
- Watch the logs for any errors
- Once you see "Uvicorn running on http://0.0.0.0:8000", it's ready

### Step 4: Get Your API URL

- Render will provide a URL like: `https://cricket-ai-analyzer.onrender.com`
- Test health check: `https://your-app.onrender.com/health`

### Step 5: Update Flutter App

Update `/cricket_ai_analyzer/lib/core/constants/api_constants.dart`:

```dart
class ApiConstants {
  // Production URL from Render
  static const String baseUrl = 'https://cricket-ai-analyzer.onrender.com';

  // For development, use:
  // static const String baseUrl = 'http://localhost:8000';

  // ... rest of the file
}
```

## Important Notes

### Model Files Size
- Total: ~36MB (18MB each)
- Ensure they're in your git repository
- If too large for git, consider Git LFS or external storage

### Free Tier Limitations
- Spins down after 15 minutes of inactivity
- First request after spin-down takes 50+ seconds
- Limited to 512MB RAM
- 750 hours/month free

### Recommended Settings for Production
- **Instance**: Starter ($7/month) minimum
- **Auto-Deploy**: Enable for automatic updates
- **Health Check**: `/health` endpoint configured

### File Storage Warning
⚠️ **Render uses ephemeral file system**:
- Uploaded videos are stored temporarily in `/tmp`
- Files are deleted when instance restarts
- For production, consider:
  - AWS S3 for file storage
  - Cloudflare R2 (cheaper alternative)
  - Return video as streaming response

### Troubleshooting

**Deployment fails**:
- Check logs in Render dashboard
- Verify requirements.txt is correct
- Ensure models/ directory exists with .pt files

**Health check fails**:
- Verify `/health` endpoint returns 200 OK
- Check if models are loading correctly

**Out of memory**:
- Upgrade to Starter instance or higher
- Models + OpenCV + video processing needs >512MB RAM

**Slow video processing**:
- Free tier has limited CPU
- Consider paid tier for faster processing
- Processing 10-second video: 30-60 seconds on free tier

## Testing Your Deployment

### Test Health Endpoint
```bash
curl https://your-app.onrender.com/health
```

Expected response:
```json
{"status": "ok"}
```

### Test Stump Detection (with curl)
```bash
curl -X POST https://your-app.onrender.com/detect_stumps \
  -F "video=@test_video.mp4"
```

### Monitor Logs
- Go to Render Dashboard → Your Service → Logs
- Watch for errors or performance issues

## Environment-Specific Configuration

### Development (Local)
```bash
export HOST=0.0.0.0
export PORT=8000
python main.py
```

### Production (Render)
- Environment variables set in Render dashboard
- Uses /tmp for temporary storage
- Automatic HTTPS enabled

## Security Considerations

1. **CORS**: Currently set to `allow_origins=["*"]` for development
   - In production, update to specific domains:
   ```python
   allow_origins=[
       "https://your-flutter-app.com",
       "https://app.your-domain.com"
   ]
   ```

2. **API Rate Limiting**: Consider adding rate limiting for production

3. **File Upload Limits**: FastAPI default is 1MB, increased for videos

## Support

For issues:
- Check Render logs first
- Verify model files are present
- Test health endpoint
- Contact support if deployment-specific issue
