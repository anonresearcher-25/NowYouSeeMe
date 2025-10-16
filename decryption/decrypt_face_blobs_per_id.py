import os
import base64
import uuid
import zlib
import msgpack
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import unpad

# --- Paths ---
METADATA_PATH = "../output/video_metadata_encrypted.msgpack"
KEYS_PATH = "../output/face_keys.msgpack"
PRIVATE_KEY_PATH = "private_key.pem"
OUTPUT_DIR = "../output/decrypted_faces"

# --- Load Private Key ---
with open(PRIVATE_KEY_PATH, "rb") as key_file:
    private_key = RSA.import_key(key_file.read())
rsa_cipher = PKCS1_OAEP.new(private_key)

# --- Load Encrypted Keys (msgpack + zlib) ---
with open(KEYS_PATH, "rb") as f:
    encrypted_keys = msgpack.unpackb(zlib.decompress(f.read()), raw=False)

# --- Decrypt AES Keys ---
face_id_to_aes_key = {}
for face_id, enc_key_dict in encrypted_keys.items():
    if face_id.startswith("video_"):
        continue
    encrypted_key = base64.b64decode(enc_key_dict["aes_key"])
    aes_key = rsa_cipher.decrypt(encrypted_key)
    face_id_to_aes_key[face_id] = aes_key

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- AES Decryption Helper ---
def decrypt_aes_bytes(ciphertext_bytes, key):
    cipher = AES.new(key, AES.MODE_ECB)
    plaintext_padded = cipher.decrypt(ciphertext_bytes)
    try:
        return unpad(plaintext_padded, 16)
    except ValueError:
        return None

# --- Load Encrypted Metadata (msgpack + zlib) ---
with open(METADATA_PATH, "rb") as f:
    encrypted_face_metadata = msgpack.unpackb(zlib.decompress(f.read()), raw=False)

# Ensure all blobs are bytes
for frame_key, blob in encrypted_face_metadata.items():
    if isinstance(blob, str):
        encrypted_face_metadata[frame_key] = blob.encode("latin1")  # preserves byte values

# --- Process Each FaceID ---
for frame_key, encrypted_blob in encrypted_face_metadata.items():
    face_id = frame_key.replace("frame_", "")
    aes_key = face_id_to_aes_key.get(face_id)
    if not aes_key:
        continue

    decrypted_bytes = decrypt_aes_bytes(encrypted_blob, aes_key)
    if decrypted_bytes is None:
        continue

    # Deserialize msgpack entries
    try:
        entries = msgpack.unpackb(decrypted_bytes, raw=False)
    except Exception as e:
        print(f"Failed to unpack entries for {face_id}: {e}")
        continue

    # Save each face image in entries
    face_dir = os.path.join(OUTPUT_DIR, face_id)
    os.makedirs(face_dir, exist_ok=True)

    for entry in entries:
        img_data_b64 = entry.get("face_image")
        if not img_data_b64:
            continue
        image_bytes = base64.b64decode(img_data_b64)

        frame_number = entry.get("frame", str(uuid.uuid4()))
        filename = f"frame_{frame_number}.png"
        filepath = os.path.join(face_dir, filename)

        with open(filepath, "wb") as img_file:
            img_file.write(image_bytes)
