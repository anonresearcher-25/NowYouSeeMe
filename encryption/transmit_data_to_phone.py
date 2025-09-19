#!/usr/bin/env python3
"""
WebSocket File Server for Android App
Sends blurred.mp4, final_matching_results.json, and video_encrypted_metadata.json
"""

import asyncio
import websockets
import json
import base64
import os
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8080
CHUNK_SIZE = 64 * 1024  # 64KB chunks
FILES_DIR = "../output"  # Directory containing the files to send

# Required files
REQUIRED_FILES = [
    "blurred.mp4",
    "face_keys.json", 
    "video_metadata_encrypted.json"
]


class FileServer:
    def __init__(self):
        self.files_dir = Path(FILES_DIR)
        self.ensure_files_directory()
        self.connected_clients = set()

    def ensure_files_directory(self):
        """Create files directory and check for required files"""
        self.files_dir.mkdir(exist_ok=True)
        
        missing_files = []
        for filename in REQUIRED_FILES:
            file_path = self.files_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            logger.warning(f"Missing files in {FILES_DIR}/ directory:")
            for filename in missing_files:
                logger.warning(f"  - {filename}")
            logger.info("Please place the required files in the files/ directory")
        else:
            logger.info("All required files found in files/ directory")
    
    async def handle_client(self, websocket, path):
        """Handle a new client connection"""
        client_address = websocket.remote_address
        logger.info(f"Client connected from {client_address}")
        self.connected_clients.add(websocket)
        
        try:
            await self.send_status(websocket, "Connected to file server")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received from {client_address}: {data}")
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {client_address}: {message}")
                    await self.send_error(websocket, "Invalid JSON message")
                except Exception as e:
                    logger.error(f"Error handling message from {client_address}: {e}")
                    await self.send_error(websocket, str(e))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_address} disconnected")
        except Exception as e:
            logger.error(f"Unexpected error with client {client_address}: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def handle_message(self, websocket, data):
        """Handle different message types from client"""
        action = data.get("action")
        
        if action == "request_files":
            files = data.get("files", REQUIRED_FILES)
            await self.send_multiple_files(websocket, files)
        
        elif action == "request_file":
            filename = data.get("fileName")
            if filename:
                await self.send_file(websocket, filename)
            else:
                await self.send_error(websocket, "Missing fileName parameter")
        elif action == "request_secure_file":
            if data.get("authToken") == self.auth_key:
                await self.send_file(websocket, "secure_data.json")
            else:
                await self.send_error(websocket, "Unauthorized: Invalid auth token")

        else:
            await self.send_error(websocket, f"Unknown action: {action}")
    
    async def send_multiple_files(self, websocket, filenames):
        """Send multiple files sequentially"""
        logger.info(f"Sending {len(filenames)} files to client")
        
        for filename in filenames:
            await self.send_file(websocket, filename)
            # Small delay between files
            await asyncio.sleep(0.1)
    
    async def send_file(self, websocket, filename):
        """Send a single file in chunks"""
        file_path = self.files_dir / filename
        
        if not file_path.exists():
            logger.warning(f"File not found: {filename}")
            await self.send_file_not_found(websocket, filename)
            return
        
        try:
            file_size = file_path.stat().st_size
            total_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            logger.info(f"Sending {filename} ({file_size} bytes) in {total_chunks} chunks")
            
            with open(file_path, 'rb') as file:
                chunk_index = 0
                
                while True:
                    chunk_data = file.read(CHUNK_SIZE)
                    if not chunk_data:
                        break
                    
                    # Encode chunk as base64
                    base64_chunk = base64.b64encode(chunk_data).decode('utf-8')
                    
                    # Send chunk
                    chunk_message = {
                        "type": "file_chunk",
                        "fileName": filename,
                        "chunkIndex": chunk_index,
                        "totalChunks": total_chunks,
                        "data": base64_chunk
                    }
                    
                    await websocket.send(json.dumps(chunk_message))
                    
                    # Progress logging
                    if chunk_index % 10 == 0 or chunk_index == total_chunks - 1:
                        progress = ((chunk_index + 1) * 100) // total_chunks
                        logger.info(f"  {filename}: {progress}% ({chunk_index + 1}/{total_chunks})")
                    
                    chunk_index += 1
                    
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)
            
            # Send completion message
            completion_message = {
                "type": "file_complete",
                "fileName": filename
            }
            await websocket.send(json.dumps(completion_message))
            
            logger.info(f"Completed sending {filename}")
        
        except Exception as e:
            logger.error(f"Error sending file {filename}: {e}")
            await self.send_error(websocket, f"Error sending {filename}: {str(e)}")
    
    async def send_status(self, websocket, message):
        """Send a status message to client"""
        status_message = {
            "type": "status",
            "message": message
        }
        await websocket.send(json.dumps(status_message))
    
    async def send_error(self, websocket, error_message):
        """Send an error message to client"""
        error_msg = {
            "type": "error",
            "message": error_message
        }
        await websocket.send(json.dumps(error_msg))
    
    async def send_file_not_found(self, websocket, filename):
        """Send file not found message"""
        not_found_msg = {
            "type": "file_not_found",
            "fileName": filename
        }
        await websocket.send(json.dumps(not_found_msg))
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on {HOST}:{PORT}")
        logger.info(f"Files directory: {self.files_dir.absolute()}")
        
        # Show available IP addresses
        self.show_server_addresses()
        
        logger.info("Waiting for client connections...")
        
        # Start the WebSocket server
        server = await websockets.serve(
            self.handle_client,
            HOST,
            PORT,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info("Server started successfully!")
        return server
    
    def show_server_addresses(self):
        """Show all available IP addresses for the server"""
        import socket
        
        logger.info("Server is accessible at these addresses:")
        
        # Get local IP addresses
        hostname = socket.gethostname()
        try:
            # Get local IP (most common case)
            local_ip = socket.gethostbyname(hostname)
            logger.info(f"  Local network: ws://{local_ip}:{PORT}/files")
        except:
            pass
        
            # Fallback method if netifaces not available
        try:
            # Connect to external server to get our IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            external_ip = s.getsockname()[0]
            s.close()
            logger.info(f"  External IP: ws://{external_ip}:{PORT}/files")
        except:
            pass
    
        logger.info(f"  Localhost: ws://127.0.0.1:{PORT}/files")
        logger.info("Use the IP address that matches your network for Android app")

def create_sample_files():
    """Create sample files for testing if they don't exist"""
    files_dir = Path(FILES_DIR)
    files_dir.mkdir(exist_ok=True)
    
    # Create sample JSON files
    sample_json = {"sample": "data", "timestamp": time.time()}
    
    json_files = [
        "face_keys.json",
        "video_metadata_encrypted.json"
    ]
    
    for filename in json_files:
        file_path = files_dir / filename
        if not file_path.exists():
            print("could not find file:", filename)
            exit(1)
    # Note about video file
    video_path = files_dir / "blurred.mp4"
    if not video_path.exists():
        logger.warning("blurred.mp4 not found. Please place your video file in the files/ directory.")
        exit(1)
        
async def main():
    """Main function to run the server"""
    # Create sample files if needed
    create_sample_files()
    
    # Create and start the file server
    file_server = FileServer()
    server = await file_server.start_server()
    
    try:
        # Keep the server running
        await server.wait_closed()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    finally:
        server.close()
        await server.wait_closed()
        logger.info("Server stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise