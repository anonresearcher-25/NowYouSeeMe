import socket
import ssl
import json
from pathlib import Path
import zlib
import msgpack

PORT = 4443
HOST = '0.0.0.0'
FILES_DIR = "../output"  # Directory containing the files to send
FILE_TO_SEND = Path(FILES_DIR) / "final_matching_results.json"

# open face_keys.json
KEYS_PATH = Path(FILES_DIR) / "face_keys.json"
with open(KEYS_PATH, "rb") as f:
    face_keys = msgpack.unpackb(zlib.decompress(f.read()), raw=False)
    AUTH_KEY = face_keys.get("video_identifier")

context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) as sock:
    # Determine local network IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"[+] Local network IP address: {local_ip}")
    print(f"[+] Access the server at: https://{local_ip}:{PORT}")

    sock.bind((HOST, PORT))
    sock.listen(5)

    with context.wrap_socket(sock, server_side=True) as ssock:
        conn, addr = ssock.accept()
        print(f"[+] Secure connection established with {addr}")

        # Receive identifier
        identifier = conn.recv(1024).decode().strip()
        print(f"[DEBUG] Received identifier: {identifier}")

        if identifier != AUTH_KEY:
            print("[-] Invalid identifier. Closing connection.")
            conn.send(b"INVALID_IDENTIFIER")
            conn.close()
        else:
            conn.send(b"IDENTIFIER_OK")
            print(f"[+] Sending file: {FILE_TO_SEND}")
            with open(FILE_TO_SEND, "rb") as f:
                while chunk := f.read(4096):
                    conn.sendall(chunk)
            print("[+] File sent successfully.")
            conn.shutdown(socket.SHUT_WR)
            conn.close()
