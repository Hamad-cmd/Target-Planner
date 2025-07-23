@echo off
echo Starting Weekly Target Planner App...
echo.
echo The app will be available at:
echo - Local: http://localhost:8501
echo - Network: http://YOUR_IP:8501
echo.
echo Share the Network URL with colleagues on the same network!
echo.
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
pause
