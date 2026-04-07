#!/bin/bash
pkill -f "streamlit run" 2>/dev/null
tmux kill-session -t titan_engine 2>/dev/null
tmux new-session -d -s titan_engine 'streamlit run app.py --server.port=8501 --server.address=0.0.0.0'
echo "========================================================"
echo " TITAN ENGINE IS LIVE IN BACKGROUND"
echo "========================================================"
echo " - Port 8501 is locked and forwarded."
echo " To view the live terminal logs at any time, type:"
echo "   tmux attach -t titan_engine"
echo " To safely exit logs without killing the app, press:"
echo "   Ctrl+B, then D"
echo "========================================================"
