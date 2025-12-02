# Production Deployment Guide

## Quick Deployment Steps

If the code is already pushed to the `main` branch, follow these steps on the production server:

### 1. SSH into Production Server
```bash
ssh user@production-server
cd /path/to/OI_Gemini_v2
```

### 2. Pull Latest Code
```bash
git pull origin main
```

### 3. Restart PM2 Process
```bash
# Check current PM2 processes
pm2 list

# Restart the application (adjust process name if different)
pm2 restart dk-OI-Gemini-v2

# OR if using a different name:
pm2 restart oi_tracker_kimi_new
```

### 4. Verify Deployment
```bash
# Check process status
pm2 status

# View recent logs
pm2 logs dk-OI-Gemini-v2 --lines 50 --nostream

# Monitor logs in real-time
pm2 logs dk-OI-Gemini-v2
```

## Automated Deployment

You can use the provided deployment script:

```bash
./deploy_production.sh
```

## Troubleshooting

### If PM2 process is not found:
```bash
# List all PM2 processes
pm2 list

# If process doesn't exist, start it:
pm2 start oi_tracker_kimi_new.py --name dk-OI-Gemini-v2 --interpreter python3

# Save PM2 configuration
pm2 save
```

### If option chain still shows N/A after restart:

1. **Check logs for errors:**
   ```bash
   pm2 logs dk-OI-Gemini-v2 --err
   ```

2. **Verify code was updated:**
   ```bash
   git log --oneline -1
   # Should show: "Fix option chain not refreshing - emit data directly from update loop"
   ```

3. **Check if update threads are running:**
   Look for log messages like:
   ```
   ✓ NSE update thread started
   ✓ BSE update thread started
   ```

4. **Verify WebSocket connection:**
   Check browser console for WebSocket connection errors

5. **Check database connection:**
   Ensure PostgreSQL is running and accessible

### Common Issues

- **Code not updated**: Make sure `git pull` completed successfully
- **Process not restarted**: PM2 may need explicit restart command
- **Python environment**: Ensure virtual environment is activated if used
- **Port conflicts**: Check if port 5050 (or configured port) is available

## Manual Restart (if PM2 fails)

```bash
# Stop the process
pm2 stop dk-OI-Gemini-v2

# Delete the process
pm2 delete dk-OI-Gemini-v2

# Start fresh
pm2 start oi_tracker_kimi_new.py --name dk-OI-Gemini-v2 --interpreter python3

# Save configuration
pm2 save
```

