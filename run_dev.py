import subprocess
import threading
import uvicorn
import platform
import time

# This function will run the frontend development server
def run_frontend():
    """
    Starts the Vite development server for the React frontend.
    """
    print("Starting frontend development server...")
    
    command = "npm.cmd run dev" if platform.system() == "Windows" else "npm run dev"
    
    # By removing stdout and stderr pipes, the output will print directly to the console.
    subprocess.Popen(
        command, 
        cwd="frontend", 
        shell=True,
    )

if __name__ == "__main__":
    # 1. Start the frontend server in a separate thread
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    frontend_thread.start()

    # Give the frontend server a moment to start up before the backend
    time.sleep(5) 

    # 2. Start the backend server in the main thread
    print("Starting backend development server on http://localhost:8000...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)