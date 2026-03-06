import os
import numpy as np
import cv2
import subprocess

def normalize_uint8(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.uint8), min_val, max_val
    normalized = (data - min_val) / (max_val - min_val) * 255.0
    return normalized.astype(np.uint8), min_val, max_val

def normalize_uint16(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.uint16), min_val, max_val
    normalized = (data - min_val) / (max_val - min_val) * (2 ** 16 - 1)
    return normalized.astype(np.uint16), min_val, max_val

def denormalize_uint8(data, min_val, max_val):
    return data / 255.0 * (max_val - min_val) + min_val

def denormalize_uint16(data, min_val, max_val):
    return data / (2 ** 16 - 1) * (max_val - min_val) + min_val

def calculate_image_size(num_points):
    image_size = 8
    while image_size * image_size < num_points:
        image_size += 8
    return image_size

def quantize_videogs_image(current_data, image_size):
    num_attributes = current_data.shape[1]
    images = {}
    min_max_info = {}
    
    for i in range(num_attributes):
        # Position attributes (0, 1, 2) -> uint16 split
        if i < 3:
            attribute_data, min_val, max_val = normalize_uint16(current_data[:, i])
            min_max_info[f'{i}_min'] = float(min_val)
            min_max_info[f'{i}_max'] = float(max_val)
            
            attribute_data_reshaped = attribute_data.reshape(-1, 1)
            image_odd = np.zeros((image_size * image_size, 1), dtype=np.uint8)
            image_even = np.zeros((image_size * image_size, 1), dtype=np.uint8)
            
            # Even = Low Byte, Odd = High Byte
            image_even[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped & 0xff)
            image_odd[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped >> 8)
            
            images[f"{2*i}"] = image_even.reshape((image_size, image_size))
            images[f"{2*i+1}"] = image_odd.reshape((image_size, image_size))
            
        else:
            attribute_data, min_val, max_val = normalize_uint8(current_data[:, i])
            min_max_info[f'{i}_min'] = float(min_val)
            min_max_info[f'{i}_max'] = float(max_val)
            
            attribute_data_reshaped = attribute_data.reshape(-1, 1)
            image = np.zeros((image_size * image_size, 1), dtype=np.uint8)
            image[:attribute_data_reshaped.shape[0], :] = attribute_data_reshaped
            
            # Offset index by +3 to match VideoGS convention (normals start at 6, etc.)
            # But wait, if we are just compressing generic attributes, we should just map i -> output_index
            # VideoGS convention: 
            # i=0 (x) -> 0, 1
            # i=1 (y) -> 2, 3
            # i=2 (z) -> 4, 5
            # i=3 (nx) -> 6
            # ...
            images[f"{i+3}"] = image.reshape((image_size, image_size))
            
    return images, min_max_info

def dequantize_videogs_image(images, frame, min_max_info):
    num_points = min_max_info[f'{frame}_num']
    
    # Determine number of attributes based on loaded images
    # Max index in images keys
    max_idx = max([int(k) for k in images.keys()])
    num_attributes = max_idx - 2 # Since last index is num_attributes-1 + 3
    
    reconstructed_data = np.zeros((num_points, num_attributes), dtype=np.float32)
    
    # Dequantize Position (0, 1, 2)
    for i in range(3):
        if f"{2*i}" in images and f"{2*i+1}" in images:
            image_even = images[f"{2*i}"].astype(np.uint16) # Low byte
            image_odd = images[f"{2*i+1}"].astype(np.uint16) # High byte
            
            image = image_even + (image_odd << 8)
            
            min_val = float(min_max_info[f'{frame}_{i}_min'])
            max_val = float(min_max_info[f'{frame}_{i}_max'])
            
            denorm = denormalize_uint16(image, min_val, max_val).flatten()[:num_points]
            reconstructed_data[:, i] = denorm
            
    # Dequantize others (3 to num_attributes-1)
    for i in range(3, num_attributes):
        img_idx = i + 3
        if f"{img_idx}" in images:
            image = images[f"{img_idx}"].astype(np.float32)
            
            min_val = float(min_max_info[f'{frame}_{i}_min'])
            max_val = float(min_max_info[f'{frame}_{i}_max'])
            
            denorm = denormalize_uint8(image, min_val, max_val).flatten()[:num_points]
            reconstructed_data[:, i] = denorm
            
    return reconstructed_data

def encode_videogs_png(images, output_path, frame_idx):
    for key, img in images.items():
        cv2.imwrite(os.path.join(output_path, f"{frame_idx}_{key}.png"), img)

def decode_videogs_png(input_folder, frame, num_attributes):
    images = {}
    # Position (0-2) -> 2 images each (0-5)
    # Attributes (3+) -> 1 image each (offset by 3)
    
    # Load position images
    for i in range(6):
        img_path = os.path.join(input_folder, f"{frame}_{i}.png")
        if os.path.exists(img_path):
            images[f"{i}"] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
    # Load other attribute images
    # We iterate until we fail to find an image, or based on expected count
    # Since we don't know exact count easily without SH degree, we can try a large range
    # or pass num_attributes if known.
    # VideoGS maps attribute i (starting from 3) to i+3.
    # Max attribute index for SH=3 is around 61. So 61+3 = 64.
    
    for i in range(6, num_attributes + 3):
        img_path = os.path.join(input_folder, f"{frame}_{i}.png")
        if os.path.exists(img_path):
            images[f"{i}"] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
    return images

def get_qp_capped_channels(sh_degree):
    """Compute channel indices that should be capped at QP=22, matching the original
    compress_image_2_video.py logic: DC color, scale, and rotation channels.

    Channel layout from compress_to_png_full_sh.py:
      0,1  = x (low, high)
      2,3  = y (low, high)
      4,5  = z (low, high)
      6    = nx
      7    = ny
      8    = nz
      9    = f_dc_0
      10   = f_dc_1
      11   = f_dc_2
      12.. = f_rest_0 .. f_rest_{n_rest-1}
      ...  = opacity
      ...  = scale_0, scale_1, scale_2
      ...  = rot_0, rot_1, rot_2, rot_3
    """
    n_rest = (sh_degree + 1) ** 2 * 3 - 3
    dc_channels = [9, 10, 11]
    scale_channels = [12 + n_rest + 1, 12 + n_rest + 2, 12 + n_rest + 3]
    rot_channels = [12 + n_rest + 4, 12 + n_rest + 5, 12 + n_rest + 6, 12 + n_rest + 7]
    return set(dc_channels + scale_channels + rot_channels)

def encode_videogs_video(frame_start, group_size, ch, ch_qp, input_group_path, output_group_path):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-start_number", str(frame_start),
        "-i", os.path.join(input_group_path, f"%d_{ch}.png"),
        "-vframes", str(group_size),
        "-c:v", "libx264",
        "-qp", str(ch_qp),
        "-pix_fmt", "yuvj444p",
        os.path.join(output_group_path, f"{ch}.mp4")
    ]
    subprocess.run(cmd, check=True)

def decode_videogs_video(frame_start, ch, input_group_path, output_group_path):
    cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", os.path.join(input_group_path, f"{ch}.mp4"),
            "-pix_fmt", "gray",
            "-start_number", str(frame_start),
            os.path.join(output_group_path, f"%d_{ch}.png")
        ]
    subprocess.run(cmd, check=True)

# ---------------------------------------------------------------------------
# Channel ↔ QP mapping
# ---------------------------------------------------------------------------

def build_channel_qp_map(sh_degree, qp):
    """Map each image channel index to its H.264 QP value.

    Capping logic (matches original compress_image_2_video.py):
      - Position MSB (1, 3, 5):     always 0 (lossless)
      - Position LSB (0, 2, 4):     qp
      - Normals (6, 7, 8):          qp
      - DC color (9, 10, 11):       min(qp, 22)
      - SH rest bands:              qp
      - Opacity:                    qp
      - Scale:                      min(qp, 22)
      - Rotation:                   min(qp, 22)
    """
    capped_qp = min(qp, 22)
    ch_map = {}

    # Position
    for i in range(3):
        ch_map[2 * i] = qp          # low byte
        ch_map[2 * i + 1] = 0       # high byte — always lossless

    # Normals
    for ch in (6, 7, 8):
        ch_map[ch] = qp

    # DC color (capped)
    for ch in (9, 10, 11):
        ch_map[ch] = capped_qp

    # SH rest bands (band l has (2l+1)*3 channels)
    ch_offset = 12
    for l in range(1, sh_degree + 1):
        band_sz = (2 * l + 1) * 3
        for j in range(band_sz):
            ch_map[ch_offset + j] = qp
        ch_offset += band_sz

    # Opacity
    ch_map[ch_offset] = qp
    ch_offset += 1

    # Scale (capped)
    for j in range(3):
        ch_map[ch_offset + j] = capped_qp
    ch_offset += 3

    # Rotation (capped)
    for j in range(4):
        ch_map[ch_offset + j] = capped_qp

    return ch_map


# ---------------------------------------------------------------------------
# Raw-video pipe helpers
# ---------------------------------------------------------------------------

def encode_channel_raw(raw_frames, image_size, ch_qp, output_mp4):
    """Pipe a sequence of grayscale uint8 frames to ffmpeg for H.264 encoding."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{image_size}x{image_size}",
        "-framerate", "25",
        "-i", "pipe:0",
        "-c:v", "libx264", "-qp", str(ch_qp),
        "-pix_fmt", "yuvj444p",
        output_mp4,
    ]
    raw_data = b"".join(frame.tobytes() for frame in raw_frames)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(input=raw_data)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed for {output_mp4}: {stderr.decode()}")


def decode_channel_raw(input_mp4, image_size, num_frames):
    """Pipe H.264 MP4 through ffmpeg and return grayscale uint8 frames."""
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", input_mp4,
        "-f", "rawvideo", "-pix_fmt", "gray",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_data, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed for {input_mp4}: {stderr.decode()}"
        )
    frame_bytes = image_size * image_size
    expected_total = frame_bytes * num_frames
    if len(stdout_data) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} bytes, got {len(stdout_data)} from {input_mp4}"
        )
    frames = []
    for i in range(num_frames):
        offset = i * frame_bytes
        frames.append(
            np.frombuffer(
                stdout_data, dtype=np.uint8, count=frame_bytes, offset=offset
            ).reshape(image_size, image_size)
        )
    return frames


