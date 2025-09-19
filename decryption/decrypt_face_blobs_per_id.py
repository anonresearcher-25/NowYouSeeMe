import os
import json
import base64
import uuid
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import unpad

# --- Paths ---
METADATA_PATH = "../output/video_metadata_encrypted.json"
KEYS_PATH = "../output/face_keys.json"
PRIVATE_KEY_PATH = "private_key.pem"
OUTPUT_DIR = "../output/decrypted_faces"

# --- Load Private Key ---
with open(PRIVATE_KEY_PATH, "rb") as key_file:
    private_key = RSA.import_key(key_file.read())
rsa_cipher = PKCS1_OAEP.new(private_key)

# --- Load Encrypted Keys ---
with open(KEYS_PATH, "r") as f:
    encrypted_keys = json.load(f)

# --- Decrypt AES Keys ---
face_id_to_aes_key = {}
for face_id, enc_key_b64 in encrypted_keys.items():
    if face_id.startswith("video_"):
            continue
    encrypted_key = base64.b64decode(enc_key_b64["aes_key"])
    aes_key = rsa_cipher.decrypt(encrypted_key)
    face_id_to_aes_key[face_id] = aes_key

# --- Load Encrypted Metadata ---
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- AES Decryption Helper ---
def decrypt_aes_base64(encrypted_b64, key):
    cipher = AES.new(key, AES.MODE_ECB)
    ciphertext = base64.b64decode(encrypted_b64)
    plaintext_padded = cipher.decrypt(ciphertext)
    try:
        return unpad(plaintext_padded, 16)
    except ValueError:
        return None

# --- Process Each Face Track Entry ---
for frame_key, encrypted_entries in metadata.items():
    for encrypted_b64 in encrypted_entries:
        # Each item was just an encrypted JSON blob — decrypt it
        # We must try every AES key until we find one that works (or pass face_id in encrypted JSON)
        for face_id, aes_key in face_id_to_aes_key.items():
            decrypted = decrypt_aes_base64(encrypted_b64, aes_key)
            if decrypted is None:
                continue
            try:
                data = json.loads(decrypted)
                if data["face_id"] != face_id:
                    continue
            except:
                continue

            # Save the image blob to disk
            img_data_b64 = data["face_image"]
            image_bytes = base64.b64decode(img_data_b64)

            face_dir = os.path.join(OUTPUT_DIR, face_id)
            os.makedirs(face_dir, exist_ok=True)

            frame_number = data.get("frame", str(uuid.uuid4()))
            filename = f"frame_{frame_number}.png"
            filepath = os.path.join(face_dir, filename)

            with open(filepath, "wb") as img_file:
                img_file.write(image_bytes)
            break  # move on to next encrypted blob once matched

print(f"✅ Decryption complete. Face images saved in: {OUTPUT_DIR}")
