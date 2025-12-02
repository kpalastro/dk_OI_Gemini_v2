#!/bin/bash
# Production deployment script for OI_Gemini_v2
# This script pulls latest code and restarts the PM2 process

set -e  # Exit on error

echo "=========================================="
echo "  OI Gemini v2 - Production Deployment"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "oi_tracker_kimi_new.py" ]; then
    echo "ERROR: Must run from project root directory"
    exit 1
fi

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "ERROR: PM2 is not installed. Install with: npm install -g pm2"
    exit 1
fi

echo "Step 1: Pulling latest code from git..."
git pull origin main

echo ""
echo "Step 2: Checking for uncommitted changes..."
if ! git diff-index --quiet HEAD --; then
    echo "WARNING: You have uncommitted changes. Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Deployment cancelled."
        exit 1
    fi
fi

echo ""
echo "Step 3: Checking PM2 process status..."
pm2 list

echo ""
echo "Step 4: Restarting PM2 process..."
# Restart the process (adjust process name if different)
pm2 restart dk-OI-Gemini-v2 || pm2 restart oi_tracker_kimi_new || {
    echo "ERROR: Could not find PM2 process. Available processes:"
    pm2 list
    echo ""
    echo "Please specify the correct process name or start it manually:"
    echo "  pm2 start oi_tracker_kimi_new.py --name dk-OI-Gemini-v2"
    exit 1
}

echo ""
echo "Step 5: Waiting for process to start..."
sleep 3

echo ""
echo "Step 6: Checking process status..."
pm2 status

echo ""
echo "Step 7: Showing recent logs..."
pm2 logs dk-OI-Gemini-v2 --lines 20 --nostream || pm2 logs oi_tracker_kimi_new --lines 20 --nostream

echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "To monitor logs in real-time:"
echo "  pm2 logs dk-OI-Gemini-v2"
echo ""
echo "To check process status:"
echo "  pm2 status"
echo ""

