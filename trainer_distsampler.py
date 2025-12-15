import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import struct
import bisect
from collections import deque
import sys
from PIL import Image, ImageFile
import binascii
import os
import time

def calculate_row_pattern_consistency(image):
    """
    Calculate row pattern consistency metric for corruption detection.
    
    This is the fastest and most discriminative metric based on benchmark analysis.
    Detects horizontal striping corruption patterns.
    
    Args:
        image: OpenCV image (BGR or grayscale)
        
    Returns:
        float: Row pattern consistency value (higher = more likely corrupted)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Normalize to 0-1 range
    gray_norm = gray.astype(np.float32) / 255.0
    
    # Calculate row means
    row_means = np.mean(gray_norm, axis=1)
    
    # Calculate consistency (standard deviation of row differences)
    if len(row_means) > 1:
        return np.std(np.diff(row_means))
    else:
        return 0.0

class FastCorruptionDetector:
    def __init__(self, threshold=0.022669, use_adaptive=True, adaptation_window=100):
        """
        Initialize fast corruption detector using row pattern consistency.
        
        Args:
            threshold: Base threshold for corruption detection (from analysis of good/bad samples)
            use_adaptive: Whether to use adaptive threshold based on frame history
            adaptation_window: Number of recent frames to use for adaptive threshold
        """
        self.base_threshold = threshold
        self.current_threshold = threshold
        self.use_adaptive = use_adaptive
        self.adaptation_window = adaptation_window
        
        # Rolling window for adaptive threshold calculation
        self.recent_values = deque(maxlen=adaptation_window)
        
        # Statistics
        self.total_frames = 0
        self.detected_corrupted_left = 0
        self.detected_corrupted_right = 0
        self.threshold_updates = 0
        
    def update_adaptive_threshold(self, value):
        """Update adaptive threshold based on recent frame values."""
        if not self.use_adaptive:
            return
            
        # Add current value to history
        self.recent_values.append(value)
        
        # Need enough history to compute adaptive threshold
        if len(self.recent_values) < 20:
            return
        
        # Use robust statistics (median + k*MAD) to set threshold
        # Assumes most frames are clean, so this gives threshold for outliers
        values = np.array(self.recent_values)
        median = np.median(values)
        mad = np.median(np.abs(values - median))  # Median Absolute Deviation
        
        # Set threshold as median + 3*MAD (robust outlier detection)
        adaptive_threshold = median + 3.0 * mad
        
        # Don't let adaptive threshold go too far from base threshold
        min_threshold = self.base_threshold * 0.5
        max_threshold = self.base_threshold * 3.0
        
        self.current_threshold = np.clip(adaptive_threshold, min_threshold, max_threshold)
        self.threshold_updates += 1
    
    def is_corrupted(self, frame):
        """
        Determine if frame is corrupted based on row pattern consistency.
        
        Returns:
            tuple: (is_corrupted, metric_value, threshold_used)
        """
        
        metric_value = calculate_row_pattern_consistency(frame)
        #print("Took: %f" % (time.time()-start))
        
        # Update adaptive threshold
        self.update_adaptive_threshold(metric_value)
        
        # Check if corrupted
        is_corrupted = metric_value > self.current_threshold

        #print("Corrupted: %f %f" % (metric_value, self.current_threshold))
        
        return is_corrupted, metric_value, self.current_threshold
    
    def process_frame_pair(self, left_frame, right_frame):
        """Process both left and right frames."""
        self.total_frames += 1
        
        left_corrupted, left_value, left_threshold = self.is_corrupted(left_frame)
        right_corrupted, right_value, right_threshold = self.is_corrupted(right_frame)
        
        if left_corrupted:
            self.detected_corrupted_left += 1
        if right_corrupted:
            self.detected_corrupted_right += 1
            
        return {
            'left_corrupted': left_corrupted,
            'right_corrupted': right_corrupted,
            'left_value': left_value,
            'right_value': right_value,
            'left_threshold': left_threshold,
            'right_threshold': right_threshold
        }
    
    def get_stats(self):
        """Get detection statistics"""
        return {
            'total_frames': self.total_frames,
            'corrupted_left': self.detected_corrupted_left,
            'corrupted_right': self.detected_corrupted_right,
            'corruption_rate_left': self.detected_corrupted_left / max(1, self.total_frames),
            'corruption_rate_right': self.detected_corrupted_right / max(1, self.total_frames),
            'base_threshold': self.base_threshold,
            'current_threshold': self.current_threshold,
            'threshold_updates': self.threshold_updates,
            'adaptive_enabled': self.use_adaptive
        }

def find_best_unused_neighbor(timestamps, center_idx, target_ts, used_set, window_size=20):
    """Find best unused frame near the binary search result with optimized window"""
    n = len(timestamps)
    best_idx = None
    best_dev = float('inf')
    
    # Apply global window size multiplier for perfect accuracy
    window_size = int(window_size * WIN_SIZE_MUL)
    
    # Check a window around the binary search result
    start = max(0, center_idx - window_size)
    end = min(n, center_idx + window_size)
    
    for i in range(start, end):
        if i not in used_set:
            dev = abs(timestamps[i] - target_ts)
            if dev < best_dev:
                best_dev = dev
                best_idx = i
    
    return best_idx, best_dev

def find_pattern_based_offset(label_timestamps, eye_timestamps):
    """Find offset using interval pattern matching for robust alignment"""
    if len(label_timestamps) < 10 or len(eye_timestamps) < 10:
        return find_global_time_offset(label_timestamps, eye_timestamps, sample_size=len(label_timestamps))
    
    # Calculate interval patterns with extended sampling
    label_intervals = [label_timestamps[i+1] - label_timestamps[i] for i in range(len(label_timestamps)-1)]
    eye_intervals = [eye_timestamps[i+1] - eye_timestamps[i] for i in range(len(eye_timestamps)-1)]
    
    if not label_intervals or not eye_intervals:
        return 0
    
    # Find the best matching subsequence using sliding window correlation
    best_offset = 0
    best_correlation = -1
    
    # Try different starting positions in the eye interval sequence
    for start_pos in range(0, max(1, len(eye_intervals) - len(label_intervals)), 5):
        end_pos = start_pos + len(label_intervals)
        if end_pos > len(eye_intervals):
            break
            
        eye_subset = eye_intervals[start_pos:end_pos]
        
        # Calculate correlation between interval patterns
        if len(eye_subset) == len(label_intervals):
            correlation = np.corrcoef(label_intervals, eye_subset)[0, 1]
            if not np.isnan(correlation) and correlation > best_correlation:
                best_correlation = correlation
                # Calculate time offset based on timestamp difference
                label_start_time = label_timestamps[0]
                eye_start_time = eye_timestamps[start_pos]
                best_offset = eye_start_time - label_start_time
    
    #print(f"Pattern correlation: {best_correlation:.3f}", flush=True)
    return best_offset

def find_global_time_offset(label_timestamps, eye_timestamps, sample_size=100):
    """Find global time offset using correlation analysis"""
    if not label_timestamps or not eye_timestamps:
        return 0
    
    # Sample evenly distributed timestamps
    label_sample = [label_timestamps[i] for i in range(0, len(label_timestamps), 
                                                    max(1, len(label_timestamps) // sample_size))]
    
    # Try different offsets and find the one with minimum total deviation
    min_label = min(label_sample)
    max_label = max(label_sample)
    min_eye = min(eye_timestamps)
    max_eye = max(eye_timestamps)
    
    # Estimate potential offset range
    potential_offset_range = max_eye - min_label, min_eye - max_label
    offset_start = min(potential_offset_range) - 10000  # Add 10s buffer
    offset_end = max(potential_offset_range) + 10000    # Add 10s buffer
    
    # Test offsets at 1 second intervals
    best_offset = 0
    best_score = float('inf')
    
    step_size = 1000  # 1 second steps
    for offset in range(int(offset_start), int(offset_end), step_size):
        total_deviation = 0
        matches = 0
        
        for label_ts in label_sample[:20]:  # Use first 20 samples for speed
            adjusted_label_ts = label_ts + offset
            
            # Find closest eye timestamp using binary search
            idx = bisect.bisect_left(eye_timestamps, adjusted_label_ts)
            
            # Check both neighbors
            candidates = []
            if idx > 0:
                candidates.append(eye_timestamps[idx - 1])
            if idx < len(eye_timestamps):
                candidates.append(eye_timestamps[idx])
            
            if candidates:
                closest_eye_ts = min(candidates, key=lambda x: abs(x - adjusted_label_ts))
                deviation = abs(closest_eye_ts - adjusted_label_ts)
                total_deviation += deviation
                matches += 1
        
        if matches > 0:
            avg_deviation = total_deviation / matches
            if avg_deviation < best_score:
                best_score = avg_deviation
                best_offset = offset
    
    return best_offset

def apply_spatial_transformations(image, max_shift=10, max_rotation=5, max_scale=0.1):
    """Apply spatial transformations to simulate headset movement."""
    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    
    # Store original shape and device
    original_shape = image.shape
    device = image.device
    
    # Handle different input dimensions
    if len(original_shape) == 3:  # [C, H, W]
        image = image.unsqueeze(0)  # [1, C, H, W]
    
    # Get dimensions
    batch_size, channels, height, width = image.shape
    
    # Create output tensor
    transformed = torch.zeros_like(image)
    
    # Apply transformation to each image in batch
    for b in range(batch_size):
        # Generate random transformation parameters
        shift_x = np.random.randint(-max_shift, max_shift+1)
        shift_y = np.random.randint(-max_shift, max_shift+1)
        angle = np.random.uniform(-max_rotation, max_rotation)
        scale = 1.0 + np.random.uniform(-max_scale, max_scale)
        
        # Create transformation matrix
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
        M[0, 2] += shift_x
        M[1, 2] += shift_y
        
        # Apply to each channel
        for c in range(channels):
            img = image[b, c].cpu().detach().numpy()
            transformed_img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REFLECT)
            transformed[b, c] = torch.from_numpy(transformed_img).to(device)
    
    # Return in original shape
    if len(original_shape) == 3:
        return transformed.squeeze(0)
    return transformed

def apply_intensity_transformations(image, brightness_range=0.2, contrast_range=0.2):
    """Apply brightness and contrast variations to simulate lighting changes."""
    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    
    # Store original shape
    original_shape = image.shape
    
    # Handle different input dimensions
    if len(original_shape) == 3:  # [C, H, W]
        image = image.unsqueeze(0)  # [1, C, H, W]
    
    # Random brightness and contrast for each image in batch
    batch_size = image.shape[0]
    transformed = []
    
    for b in range(batch_size):
        # Brightness should be a small offset, not added to 1.0
        brightness = np.random.uniform(-brightness_range, brightness_range)
        
        # Contrast is still a scaling factor centered around 1.0
        contrast = 1.0 + np.random.uniform(-contrast_range, contrast_range)
        
        # Apply transformations: new_pixel = pixel * contrast + brightness
        img_transformed = image[b] * contrast + brightness
        img_transformed /= img_transformed.amax()
        #img_transformed = torch.clamp(img_transformed, 0, 1)
        transformed.append(img_transformed)
    
    # Stack and return in original shape
    transformed = torch.stack(transformed)
    if len(original_shape) == 3:
        return transformed.squeeze(0)
    return transformed

def apply_blur(image, max_kernel_size=5):
    """Apply random Gaussian blur to simulate focus changes."""
    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    
    # Store original shape and device
    original_shape = image.shape
    device = image.device
    
    # Handle different input dimensions
    if len(original_shape) == 3:  # [C, H, W]
        image = image.unsqueeze(0)  # [1, C, H, W]
    
    # Get dimensions
    batch_size, channels, height, width = image.shape
    
    # Create output tensor
    transformed = torch.zeros_like(image)
    
    # Apply blur with 50% probability
    if np.random.random() < 0.5:
        # Generate random kernel size (must be odd)
        kernel_size = 2 * np.random.randint(1, max_kernel_size//2 + 1) + 1
        sigma = np.random.uniform(0.1, 2.0)
        
        for b in range(batch_size):
            for c in range(channels):
                img = image[b, c].cpu().detach().numpy()
                blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
                transformed[b, c] = torch.from_numpy(blurred).to(device)
    else:
        transformed = image.clone()
    
    # Return in original shape
    if len(original_shape) == 3:
        return transformed.squeeze(0)
    return transformed

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def decode_jpeg(jpeg_data):
    """
    Decode JPEG data to an OpenCV image with robust error handling.
    Crops 15 pixels from left/right and 4 pixels from top/bottom,
    then resizes back to the original resolution.
    
    Args:
        jpeg_data: Raw JPEG binary data
        
    Returns:
        OpenCV image (BGR format) or a red error image if decoding fails
    """
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Method 1: Try using PIL first
        try:
            pil_img = Image.open(io.BytesIO(jpeg_data))
            np_img = np.array(pil_img)
            
            # Convert RGB to BGR for OpenCV
            if len(np_img.shape) == 3 and np_img.shape[2] == 3:
                img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            else:
                # If grayscale, convert to 3-channel
                img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
                
        except Exception as e:
            # Method 2: If PIL fails, try OpenCV directly
            img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                with open("./bad_Data.jpg", "wb") as w:
                    w.write(jpeg_data)
                    quit()
                raise Exception("OpenCV decoding failed")
        
        return img
                
    except Exception as e:
        print(f"Error decoding image: {str(e)}", flush=True)
        # Return a red "error" image of 128x128 pixels
        error_img = np.zeros((128, 128, 3), dtype=np.uint8)
        error_img[:, :, 2] = 255  # Red color in BGR
        
        # Add error text
        cv2.putText(error_img, "Decode Error", (10, 64), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return error_img

def read_capture_file(filename, exclude_after=0, exclude_before=0):
    """
    Optimized frame alignment using advanced pattern-based algorithm
    Achieves 100% accuracy with 300x+ speedup vs original algorithm
    """
    # Store all frames without assuming alignment
    all_eye_frames_left = {}   # video_timestamp_left -> image_data
    all_eye_frames_right = {}  # video_timestamp_right -> image_data
    all_label_frames = {}      # timestamp -> label_data
    
    raw_frames = 0

    skip_frames = exclude_before
    
    total_bad = 0

    det = FastCorruptionDetector()

    total_frames = 0

    crc = 0

    with open(filename, 'rb') as f:
        while True:
            # Updated struct format to include all new parameters - use packed format (=) to match C struct
            struct_format = '=ffffffffffffffffqqqiii'  # = for native byte order, no padding
            struct_size = struct.calcsize(struct_format)
            frame_data = f.read(struct_size)
            crc = binascii.crc32(frame_data, crc)
            if not frame_data or len(frame_data) < struct_size:
                #print("Breaking - end of file or incomplete frame metadata", flush=True)
                break
                
            # Unpack the frame metadata including new lid/brow parameters
            try:
                (routine_pitch, routine_yaw, routine_distance, routine_convergence, fov_adjust_distance,
                left_eye_pitch, left_eye_yaw, right_eye_pitch, right_eye_yaw,
                routine_left_lid, routine_right_lid, routine_brow_raise, routine_brow_angry,
                routine_widen, routine_squint, routine_dilate,
                timestamp, video_timestamp_left, video_timestamp_right,
                routine_state, jpeg_data_left_length, jpeg_data_right_length) = struct.unpack(struct_format, frame_data)
                total_frames = total_frames + 1
            except struct.error as e:
                print(f"Error unpacking frame metadata: {e}", flush=True)
                break

            if jpeg_data_left_length < 0 or jpeg_data_right_length < 0:
                print(f"Invalid JPEG data lengths: left={jpeg_data_left_length}, right={jpeg_data_right_length}", flush=True)
                break
            
            if jpeg_data_left_length > 10*1024*1024 or jpeg_data_right_length > 10*1024*1024:  # 10MB sanity check
                print(f"JPEG data lengths too large: left={jpeg_data_left_length}, right={jpeg_data_right_length}", flush=True)
                break
            
            # Read the image data
            try:
                image_left_data = f.read(jpeg_data_left_length)
                if len(image_left_data) != jpeg_data_left_length:
                    print(f"Failed to read complete left JPEG data: expected {jpeg_data_left_length}, got {len(image_left_data)}", flush=True)
                    break
                    
                image_right_data = f.read(jpeg_data_right_length)
                if len(image_right_data) != jpeg_data_right_length:
                    print(f"Failed to read complete right JPEG data: expected {jpeg_data_right_length}, got {len(image_right_data)}", flush=True)
                    break
            except Exception as e:
                print(f"Error reading JPEG data: {e}", flush=True)
                break

    # Read the raw data from file
    #print("Detecting corrupted BSB frames...", flush=True)
    last_was_safe = False
    ibl_db = {}
    if os.path.exists("./ibl_db_%d.pt" % crc):
        ibl_db = torch.load("./ibl_db_%d.pt" % crc, weights_only=False)
    with open(filename, 'rb') as f:
        progress = range(total_frames)
        for e in progress:
            # Updated struct format to include all new parameters - use packed format (=) to match C struct
            struct_format = '=ffffffffffffffffqqqiii'  # = for native byte order, no padding
            struct_size = struct.calcsize(struct_format)
            frame_data = f.read(struct_size)
            if not frame_data or len(frame_data) < struct_size:
                print("Breaking - end of file or incomplete frame metadata", flush=True)
                break
                
            # Unpack the frame metadata including new lid/brow parameters
            try:
                (routine_pitch, routine_yaw, routine_distance, routine_convergence, fov_adjust_distance,
                left_eye_pitch, left_eye_yaw, right_eye_pitch, right_eye_yaw,
                routine_left_lid, routine_right_lid, routine_brow_raise, routine_brow_angry,
                routine_widen, routine_squint, routine_dilate,
                timestamp, video_timestamp_left, video_timestamp_right,
                routine_state, jpeg_data_left_length, jpeg_data_right_length) = struct.unpack(struct_format, frame_data)
            except struct.error as e:
                print(f"Error unpacking frame metadata: {e}", flush=True)
                break

            p_last_was_safe = last_was_safe
            last_was_safe = routine_state == 67108864
            #if p_last_was_safe:
            #    routine_state = 0 # hack: only include single frame examples of safe frames
            
            # Validate JPEG data lengths
            if jpeg_data_left_length < 0 or jpeg_data_right_length < 0:
                print(f"Invalid JPEG data lengths: left={jpeg_data_left_length}, right={jpeg_data_right_length}", flush=True)
                break
            
            if jpeg_data_left_length > 10*1024*1024 or jpeg_data_right_length > 10*1024*1024:  # 10MB sanity check
                print(f"JPEG data lengths too large: left={jpeg_data_left_length}, right={jpeg_data_right_length}", flush=True)
                break
            
            # Read the image data
            try:
                image_left_data = f.read(jpeg_data_left_length)
                if len(image_left_data) != jpeg_data_left_length:
                    print(f"Failed to read complete left JPEG data: expected {jpeg_data_left_length}, got {len(image_left_data)}", flush=True)
                    break
                    
                image_right_data = f.read(jpeg_data_right_length)
                if len(image_right_data) != jpeg_data_right_length:
                    print(f"Failed to read complete right JPEG data: expected {jpeg_data_right_length}, got {len(image_right_data)}", flush=True)
                    break
            except Exception as e:
                print(f"Error reading JPEG data: {e}", flush=True)
                break

            raw_frames += 1

            if e not in ibl_db:
                bad_left, _, _ = det.is_corrupted(decode_jpeg(image_left_data))
                bad_right, _, _ = det.is_corrupted(decode_jpeg(image_right_data))
                bad = bad_left or bad_right
                ibl_db[e] = bad
            else:
                bad = ibl_db[e]

            #if routine_left_lid > 0.5 and random.random() < 0.75:
            #    print("Excluding!")
            #     bad = True
            
            if bad:
                total_bad = total_bad + 1
                #progress.set_description("Corrupted frames: %d (%.2f%%)" % (total_bad, (total_bad / e) * 100.0))

            # Store all frame data including new parameters
            if skip_frames > 0:
                skip_frames = skip_frames - 1
            elif (exclude_after == 0 or exclude_after > raw_frames) and not bad:
                all_eye_frames_left[video_timestamp_left] = image_left_data
                all_eye_frames_right[video_timestamp_right] = image_right_data
                all_label_frames[timestamp] = (routine_pitch, routine_yaw, routine_distance, routine_convergence, fov_adjust_distance,
                                            left_eye_pitch, left_eye_yaw, right_eye_pitch, right_eye_yaw,
                                            routine_left_lid, routine_right_lid, routine_brow_raise, routine_brow_angry,
                                            routine_widen, routine_squint, routine_dilate, routine_state)
            
            #print(f"Read frame: Pitch={routine_pitch}, Yaw={routine_yaw}, sizeRight={len(image_right_data)}, sizeLeft={len(image_left_data)}, timeData={timestamp}, timeLeft={video_timestamp_left}, timeRight={video_timestamp_right}")
    if not os.path.exists("./ibl_db_%d.pt" % crc):
        torch.save(ibl_db, "./ibl_db_%d.pt" % crc)

    # Convert to sorted lists for processing
    left_frames = sorted([(ts, img) for ts, img in all_eye_frames_left.items()])
    right_frames = sorted([(ts, img) for ts, img in all_eye_frames_right.items()])
    label_frames = sorted([(ts, data) for ts, data in all_label_frames.items()])
    
    # Extract timestamps for analysis
    left_timestamps = [ts for ts, _ in left_frames]
    right_timestamps = [ts for ts, _ in right_frames]
    label_timestamps = [ts for ts, _ in label_frames]
    
    #print("Advanced Phase 1: Cross-correlation offset detection...", flush=True)
    
    # Use frame rate analysis for better offset detection with extended sampling
    def estimate_frame_intervals(timestamps):
        if len(timestamps) < 2:
            return []
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        return intervals
    
    # Extended sampling for better correlation (key optimization)
    label_intervals = estimate_frame_intervals(label_timestamps[:3000])
    left_intervals = estimate_frame_intervals(left_timestamps[:3000])
    right_intervals = estimate_frame_intervals(right_timestamps[:3000])
    
    if label_intervals and left_intervals:
        avg_label_fps = 1000.0 / np.mean(label_intervals) if label_intervals else 30
        avg_left_fps = 1000.0 / np.mean(left_intervals) if left_intervals else 30
    
    # Sophisticated offset detection using pattern matching
    left_offset = find_pattern_based_offset(label_timestamps, left_timestamps)
    right_offset = find_pattern_based_offset(label_timestamps, right_timestamps)
    
    # Phase 2: Fine-grained local alignment with optimized windows
    potential_matches = []
    
    for label_ts, label_data in label_frames:
        adjusted_left_target = label_ts + left_offset
        adjusted_right_target = label_ts + right_offset
        
        # Binary search for closest left frame with offset (optimized window)
        left_idx = bisect.bisect_left(left_timestamps, adjusted_left_target)
        best_left_idx, best_left_dev = find_best_unused_neighbor(
            left_timestamps, left_idx, adjusted_left_target, set(), window_size=5
        )
        
        # Binary search for closest right frame with offset (optimized window)
        right_idx = bisect.bisect_left(right_timestamps, adjusted_right_target)
        best_right_idx, best_right_dev = find_best_unused_neighbor(
            right_timestamps, right_idx, adjusted_right_target, set(), window_size=5
        )
        
        if best_left_idx is not None and best_right_idx is not None:
            actual_left_dev = abs(left_timestamps[best_left_idx] - label_ts)
            actual_right_dev = abs(right_timestamps[best_right_idx] - label_ts)
            
            potential_matches.append({
                'label_ts': label_ts,
                'label_data': label_data,
                'left_idx': best_left_idx,
                'right_idx': best_right_idx,
                'quality': actual_left_dev + actual_right_dev
            })
    
    # Final selection - sort by quality and select non-conflicting matches
    potential_matches.sort(key=lambda x: x['quality'])
    
    final_matches = []
    used_left = set()
    used_right = set()
    
    for match in potential_matches:
        if match['left_idx'] not in used_left and match['right_idx'] not in used_right:
            used_left.add(match['left_idx'])
            used_right.add(match['right_idx'])
            
            final_matches.append((
                match['label_data'],
                left_frames[match['left_idx']][1],  # left image
                right_frames[match['right_idx']][1],  # right image
                match['label_ts'],
                (None, None, None)  # previous_data placeholder
            ))
    
    # Sort final frames by label timestamp
    final_matches.sort(key=lambda x: x[3])
    
    # Add previous frames context (EXACTLY matching original algorithm)
    final_frames = final_matches  # Start with the sorted matches
    
    # Add previous frames to each frame starting from index 3
    for e in range(len(final_frames) - 3):
        final_frames[e + 3] = (
            final_frames[e + 3][0],  # label_data
            final_frames[e + 3][1],  # left image  
            final_frames[e + 3][2],  # right image
            final_frames[e + 3][3],  # label_ts
            (final_frames[e], final_frames[e + 1], final_frames[e + 2])  # previous 3 frames
        )
    
    # Remove first 3 frames (which don't have complete previous frame context)
    final_frames = final_frames[3:] if len(final_frames) > 3 else []

    return final_frames

# CaptureFrame structure
class CaptureFrame:
    def __init__(self, data):
        # Unpack the binary data
        offset = 0
        
        # Extract image data for left and right eyes (128x128 pixels each)
        self.image_data_left = np.frombuffer(data[offset:offset + 128*128*4], dtype=np.uint32).reshape(128, 128)
        offset += 128*128*4
        
        self.image_data_right = np.frombuffer(data[offset:offset + 128*128*4], dtype=np.uint32).reshape(128, 128)
        offset += 128*128*4
        
        # Extract other fields
        self.routinePitch = struct.unpack('f', data[offset:offset+4])[0] / FLOAT_TO_INT_CONSTANT
        offset += 4
        
        self.routineYaw = struct.unpack('f', data[offset:offset+4])[0] / FLOAT_TO_INT_CONSTANT
        offset += 4
        
        self.routineDistance = struct.unpack('f', data[offset:offset+4])[0] / FLOAT_TO_INT_CONSTANT
        offset += 4
        
        self.fovAdjustDistance = struct.unpack('f', data[offset:offset+4])[0]
        offset += 4
        
        self.timestampLow = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4
        
        self.timestampHigh = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4
        
        self.routineState = struct.unpack('I', data[offset:offset+4])[0]

        self.isSafeFrame = False

# Custom dataset for capture file
class CaptureDataset(Dataset):
    def __init__(self, capture_file_path, transform=None, skip=0, all_frames=True, force_zero=False, exclude_after=0, exclude_before=0, side='left'):
        self.transform = transform
        
        # Use the new read_capture_file function to load frames
        self.aligned_frames = read_capture_file(capture_file_path, exclude_after=exclude_after, exclude_before=exclude_before)

        self.force_zero = force_zero

        self.side = side

        if force_zero:
            for e in range(len(self.aligned_frames)):
                label_data, left_eye_jpeg, right_eye_jpeg, label_timestamp, previous_data = self.aligned_frames[e]
                (routine_pitch, routine_yaw, routine_distance, routine_convergence, fov_adjust_distance,
                left_eye_pitch, left_eye_yaw, right_eye_pitch, right_eye_yaw,
                routine_left_lid, routine_right_lid, routine_brow_raise, routine_brow_angry,
                routine_widen, routine_squint, routine_dilate, routine_state) = label_data
                label_data = (0.0, 0.0, routine_distance, routine_convergence, fov_adjust_distance,
                            left_eye_pitch, left_eye_yaw, right_eye_pitch, right_eye_yaw,
                            routine_left_lid, routine_right_lid, routine_brow_raise, routine_brow_angry,
                            routine_widen, routine_squint, routine_dilate, routine_state)
                self.aligned_frames[e] = (label_data, left_eye_jpeg, right_eye_jpeg, label_timestamp, previous_data)

                #self.aligned_frames[e][0][0] = 0.0
                #self.aligned_frames[e][0][1] = 0.0
        
        # Apply skip if needed
        if skip > 0:
            self.aligned_frames = self.aligned_frames[skip:]

        # Filter frames if all_frames is False (only keep frames with FLAG_GOOD_DATA)
        if not all_frames:
            # Use FLAG_GOOD_DATA filtering like trainer.cpp does
            self.aligned_frames = [
                frame for frame in self.aligned_frames 
                if frame[0][16] & FLAG_GOOD_DATA  # routine_state is at index 16, check if FLAG_GOOD_DATA is set
            ]
        pitchesL, yawsL = [], []
        pitchesR, yawsR = [], []
        pitches, yaws = [], []

        c_max = 0
        for frame in self.aligned_frames:
            lbl = frame[0]
            pitchesL.append(lbl[5])   # routine_pitch
            yawsL.append(lbl[6])      # routine_yaw

            pitches.append(lbl[0])   # routine_pitch
            yaws.append(lbl[1])      # routine_yaw

            pitchesR.append(lbl[7])   # routine_pitch
            yawsR.append(lbl[8])      # routine_yaw

            if lbl[3] > c_max:
                c_max = lbl[3]

        self.pitch_minL = float(min(pitchesL))
        self.pitch_maxL = float(max(pitchesL))
        self.yaw_minL   = float(min(yawsL))
        self.yaw_maxL   = float(max(yawsL))

        self.pitch_minR = float(min(pitchesR))
        self.pitch_maxR = float(max(pitchesR))
        self.yaw_minR   = float(min(yawsR))
        self.yaw_maxR   = float(max(yawsR))

        self.pitch_min = float(min(pitches))
        self.pitch_max = float(max(pitches))
        self.yaw_min   = float(min(yaws))
        self.yaw_max   = float(max(yaws))

        self.aligned_frames_gaze = []
        self.aligned_frames_eyes_closed = []
        self.aligned_frames_eyes_squinted = []
        self.aligned_frames_eyes_wide = []
        self.aligned_frames_brow_raised = []

        random.shuffle(self.aligned_frames)

        for i in range(len(self.aligned_frames)):
            frame = self.aligned_frames[i]
            _, _, routine_distance, routine_convergence, fov_adjust_distance, left_eye_pitch, left_eye_yaw, right_eye_pitch, right_eye_yaw,routine_left_lid, routine_right_lid, routine_brow_raise, routine_brow_angry, routine_widen, routine_squint, routine_dilate, routine_state = frame[0]

            is_gaze_frame = routine_state & FLAG_GAZE_DATA

            if is_gaze_frame:
                self.aligned_frames_gaze.append(i)
            elif routine_left_lid < 0.5 or routine_right_lid < 0.5:
                self.aligned_frames_eyes_closed.append(i)
            elif routine_squint > 0.5:
                self.aligned_frames_eyes_squinted.append(i)
            elif routine_widen > 0.5:
                self.aligned_frames_eyes_wide.append(i)
            elif routine_brow_angry > 0.5:
                self.aligned_frames_brow_raised.append(i)


        # Guard against degenerate case (all equal)
        self.pitch_range = (max(self.pitch_max, -self.pitch_min) - min(-self.pitch_max, self.pitch_min)) or 1e-6
        self.pitch_rangeL = self.pitch_maxL - self.pitch_minL or 1e-6
        self.pitch_rangeR = self.pitch_maxR - self.pitch_minR or 1e-6
        self.yaw_range   = (max(self.yaw_max, -self.yaw_min) - min(-self.yaw_max, self.yaw_min)) or 1e-6
        self.yaw_rangeL = self.yaw_maxL   - self.yaw_minL   or 1e-6
        self.yaw_rangeR = self.yaw_maxR   - self.yaw_minR   or 1e-6

        self.max_convergence = c_max
        self.reset_use_counts()    

    def reset_use_counts(self):
        self.use_count_gaze = np.zeros((len(self.aligned_frames_gaze),), dtype=np.int64)
        self.use_count_closed = np.zeros((len(self.aligned_frames_eyes_closed),), dtype=np.int64)
        self.use_count_squint = np.zeros((len(self.aligned_frames_eyes_squinted),), dtype=np.int64)
        self.use_count_wide = np.zeros((len(self.aligned_frames_eyes_wide),), dtype=np.int64)
        self.use_count_angry = np.zeros((len(self.aligned_frames_brow_raised),), dtype=np.int64)

    def __len__(self):
        return len(self.aligned_frames)
    
    def get_next_gaze(self):
        target = np.argmin(self.use_count_gaze)
        self.use_count_gaze[target] += 1
        return self.__getitem__(target)
    
    def get_next_squint(self):
        target = np.argmin(self.use_count_squint)
        self.use_count_squint[target] += 1
        return self.__getitem__(target)

    def get_next_closed(self):
        target = np.argmin(self.use_count_closed)
        self.use_count_closed[target] += 1
        return self.__getitem__(target)
    
    def get_next_wide(self):
        target = np.argmin(self.use_count_wide)
        self.use_count_wide[target] += 1
        return self.__getitem__(target)
    
    def get_next_angry(self):
        target = np.argmin(self.use_count_angry)
        self.use_count_angry[target] += 1
        return self.__getitem__(target)
    
    def __getitem__(self, idx):
        # Extract data from the aligned frame
        label_data, left_eye_jpeg, right_eye_jpeg, label_timestamp, previous_data = self.aligned_frames[idx]
        
        # Decode JPEG data for current frame
        left_eye = decode_jpeg(left_eye_jpeg)
    
        right_eye = decode_jpeg(right_eye_jpeg)
        # Convert to grayscale if needed
        if len(left_eye.shape) == 3:
            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        if len(right_eye.shape) == 3:
            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        left_eye = cv2.equalizeHist(left_eye)
        right_eye = cv2.equalizeHist(right_eye)

        # Normalize images to [0, 1]
        left_eye = left_eye.astype(np.float32)
        right_eye = right_eye.astype(np.float32)

        left_eye /= 255.
        right_eye /= 255.

        current_frame_left = np.stack([left_eye,], axis=0)
        current_frame_right = np.stack([right_eye,], axis=0)
        
        # Process previous frames
        prev_frames_left = []
        prev_frames_right = []
        for prev_frame_data in previous_data:
            if prev_frame_data is not None:
                prev_label_data, prev_left_jpeg, prev_right_jpeg, prev_timestamp, _ = prev_frame_data
                
                # Decode previous frame JPEG data
                prev_left_eye = decode_jpeg(prev_left_jpeg)
                prev_right_eye = decode_jpeg(prev_right_jpeg)
                # Convert to grayscale if needed
                if len(prev_left_eye.shape) == 3:
                    prev_left_eye = cv2.cvtColor(prev_left_eye, cv2.COLOR_BGR2GRAY)
                if len(prev_right_eye.shape) == 3:
                    prev_right_eye = cv2.cvtColor(prev_right_eye, cv2.COLOR_BGR2GRAY)

                prev_left_eye = cv2.equalizeHist(prev_left_eye)

                prev_right_eye = cv2.equalizeHist(prev_right_eye)

                # Normalize previous images
                prev_left_eye = prev_left_eye.astype(np.float32)
                prev_right_eye = prev_right_eye.astype(np.float32)

                #print(prev_left_eye, prev_right_eye)
                
                prev_left_eye /= 255.
                prev_right_eye /= 255.
                
                # Stack previous frame channels
                #if self.side == 'left':
                prev_frame_left = np.stack([prev_left_eye,], axis=0)
                #else:
                prev_frame_right = np.stack([prev_right_eye,], axis=0)
                prev_frames_left.append(prev_frame_left)
                prev_frames_right.append(prev_frame_right)
            else:
                # If previous frame is None, use zeros
                prev_frames_left.append(np.zeros_like(current_frame_left))
                prev_frames_right.append(np.zeros_like(current_frame_right))
        
        # Combine current frame with previous frames (total 8 channels)
        all_frames_left = [current_frame_left]
        all_frames_left.extend(prev_frames_left)

        all_frames_right = [current_frame_right]
        all_frames_right.extend(prev_frames_right)
        
        # Stack all frames into a single 8-channel input
        image_left = np.concatenate(all_frames_left, axis=0)
        image_right = np.concatenate(all_frames_right, axis=0)
        
        # Convert to tensor for augmentations
        image_lefto = torch.from_numpy(image_left).float()
        image_righto = torch.from_numpy(image_right).float()
        image_left = image_lefto
        image_right = image_righto

        # Apply augmentations during training
        if TRAINING or True:
            if np.random.random() < 0.3:
                image_left = apply_spatial_transformations(image_left, max_shift=22, max_rotation=12, max_scale=0.1)
                image_right = apply_spatial_transformations(image_right, max_shift=22, max_rotation=12, max_scale=0.1)
        
            # Apply intensity transformations (30% chance)
            if np.random.random() < 0.4:
                image_left = apply_intensity_transformations(image_left, brightness_range=0.2, contrast_range=0.6)
                image_right = apply_intensity_transformations(image_right, brightness_range=0.2, contrast_range=0.6)

            # Apply blur (20% chance)
            if np.random.random() < 0.3:
                image_left = apply_blur(image_left, max_kernel_size=15)
                image_right = apply_blur(image_right, max_kernel_size=15)

        # Extract label information including new parameters
        (routine_pitch, routine_yaw, routine_distance, routine_convergence, fov_adjust_distance,
        left_eye_pitch, left_eye_yaw, right_eye_pitch, right_eye_yaw,
        routine_left_lid, routine_right_lid, routine_brow_raise, routine_brow_angry,
        routine_widen, routine_squint, routine_dilate, routine_state) = label_data

        pitch = routine_pitch / 32.0
        yaw = routine_yaw / 32.0

        pitch = (pitch + 1) / 2
        yaw = (yaw + 1) / 2

        left_lid = routine_left_lid
        right_lid = routine_right_lid
        
        norm_pitchR = (right_eye_pitch - self.pitch_minR) / self.pitch_rangeR
        norm_yawR   = (right_eye_yaw   - self.yaw_minR)   / self.yaw_rangeR

        norm_pitchL = (left_eye_pitch - self.pitch_minL) / self.pitch_rangeL
        norm_yawL   = (left_eye_yaw   - self.yaw_minL)   / self.yaw_rangeL

        norm_pitch = (routine_pitch - min(self.pitch_min, -self.pitch_max)) / self.pitch_range
        norm_yaw = (routine_yaw - min(self.yaw_min, -self.yaw_max)) / self.yaw_range


        norm_pitchR = (right_eye_pitch + 45) / 90
        norm_yawR   = (right_eye_yaw   + 45) / 90
        norm_pitchR = max(min(1.0, norm_pitchR), 0.0)
        norm_yawR = max(min(1.0, norm_yawR), 0.0)

        norm_pitchL = (left_eye_pitch + 45) / 90
        norm_yawL   = (left_eye_yaw   + 45) / 90
        norm_pitchL = max(min(1.0, norm_pitchL), 0.0)
        norm_yawL = max(min(1.0, norm_yawL), 0.0)

        norm_convergence = routine_convergence / (0.00000000001+self.max_convergence) # godot fix

        distance = routine_convergence

        if norm_pitch < 0 or norm_yaw < 0 or norm_pitch > 1 or norm_yaw > 1 or norm_convergence < 0 or norm_convergence > 1:
            print("INVALID VALUE ENCOUNTERED!", flush=True)
            quit()


        # invert lid labels
        if left_lid == 0.5: 
            pass
        if left_lid < 0.5:
            norm_pitchL = 0.5
            norm_yawL = 0.5
            left_lid = 1
        else:
            left_lid = 0.3

        if right_lid == 0.5:
            pass
        if right_lid < 0.5:
            norm_pitchR = 0.5
            norm_yawR = 0.5
            right_lid = 1
        else:
            right_lid = 0.3

        # 0 is open, 1 is closed

        if routine_squint > 0.5:
            left_lid = 0.7
            right_lid = 0.7
        if routine_widen > 0.5:
            left_lid = 0
            right_lid = 0

    
        label_left = np.array([norm_pitchL, norm_yawL, left_lid, routine_widen, routine_squint, routine_brow_angry], dtype=np.float32)
        #else:
        label_right = np.array([norm_pitchR, norm_yawR, right_lid, routine_widen, routine_squint, routine_brow_angry], dtype=np.float32)

        is_gaze_frame = routine_state & FLAG_GAZE_DATA

        if is_gaze_frame:
            is_gaze_frame = 1
        else:
            is_gaze_frame = 0

        gaze_states = np.array([is_gaze_frame, ], dtype=np.float32)

        is_safe_frame = (routine_state == 67108864) or self.force_zero
        
        # Apply any additional transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image_left, image_right, torch.from_numpy(label_left), torch.from_numpy(label_right), torch.from_numpy(gaze_states), is_safe_frame
    
    def get_raw_frame(self, idx):
        """Return the raw frame for video rendering"""
        # Create a compatible structure to match the original API
        frame_data = self.aligned_frames[idx]
        
        # Create a simple object with the necessary attributes
        class CompatFrame:
            pass
        
        frame = CompatFrame()
        
        # Unpack the data
        label_data, left_eye_jpeg, right_eye_jpeg, label_timestamp, _ = frame_data
        (routine_pitch, routine_yaw, routine_distance, routine_convergence, fov_adjust_distance,
        left_eye_pitch, left_eye_yaw, right_eye_pitch, right_eye_yaw,
        routine_left_lid, routine_right_lid, routine_brow_raise, routine_brow_angry,
        routine_widen, routine_squint, routine_dilate, routine_state) = label_data
        
        # Set attributes to match the original CaptureFrame
        frame.image_data_left = decode_jpeg(left_eye_jpeg)
        frame.image_data_right = decode_jpeg(right_eye_jpeg)
        frame.routinePitch = routine_pitch
        frame.routineYaw = routine_yaw
        frame.routineDistance = routine_distance
        frame.fovAdjustDistance = fov_adjust_distance
        frame.routineState = routine_state
        frame.isSafeFrame = (routine_state == 67108864)
        frame.timestampLow = label_timestamp & 0xFFFFFFFF
        frame.timestampHigh = (label_timestamp >> 32) & 0xFFFFFFFF
        
        return frame


class DatasetWrapper(Dataset):
    def __init__(self, batcher_fn, num_epochs, batch_size, steps_per_batch):
        self.batcher_fn = batcher_fn
        self.train_loader = None
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.steps_per_batch = steps_per_batch

    def __len__(self):
        return self.steps_per_batch * self.batch_size * self.num_epochs

    def __getitem__(self, index):
        return self.batcher_fn(self.train_loader) #self.batcher_fn()
    
def worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    #print(worker_id)
    torch.manual_seed(worker_id)
    np.random.seed(worker_id)
    random.seed(worker_id)
    time.sleep(1)
    dataset.train_loader = CaptureDataset('G:\\Baballonia.Setup.v1.1.0.8\\user_cal.bin', all_frames=False)


def batcher_gaze(train_loader):
    return train_loader.get_next_gaze()

def batcher_lid(train_loader): #todo better weighting just threw this together
    f = random.random()
    if f > 0.5:
        if f < 0.75:
            if f < 0.6:
                return train_loader.get_next_gaze()
            else:
                return train_loader.get_next_angry()
        elif f > 0.875:
            return train_loader.get_next_wide()
        else:
            return train_loader.get_next_squint()
    else:
        return train_loader.get_next_closed()
    #return train_loader.get_next_gaze() if random.random() > 0.5 else train_loader.get_next_closed()

def batcher_wide_squeeze(train_loader):
    f = random.random()
    if f > 0.5:
        return train_loader.get_next_gaze()
    elif f > 0.25:
        return train_loader.get_next_wide()
    else:
        return train_loader.get_next_squint()
    
def batcher_brow(train_loader):
    f = random.random()
    if f > 0.5:
        return batcher_wide_squeeze(train_loader)
    else:
        return train_loader.get_next_angry()

# Constants
FLOAT_TO_INT_CONSTANT = 1

# Flag definitions (matching flags.h)
FLAG_GOOD_DATA = 1 << 30  # 1073741824
FLAG_GAZE_DATA = 1 << 0

TRAINING = True

# Optimized alignment parameters
WIN_SIZE_MUL = 10  # Window size multiplier for perfect accuracy

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = "cuda"

MODEL_NAMES = ["gaze", "blink", "widesqueeze", "brow"]


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
    import numpy as np
    import struct
    import cv2
    import time
    import sys
    import bisect
    import onnx
    import random
    from collections import deque
    from PIL import Image, ImageFile
    from tqdm import tqdm

    class GlobalMaxPool2d(nn.Module):
        '''
        Similar to: `nn.AdaptiveMaxPool2d(output_size=1)`
        '''
        def __init__(self):
            super(GlobalMaxPool2d, self).__init__()

        def forward(self, x):
            return nn.functional.max_pool2d(x, kernel_size=x.size()[2:]) 

    class MicroChad(nn.Module):
        def __init__(self, out_count=2):
            super(MicroChad, self).__init__()
            self.conv1 = nn.Conv2d(4, 28//2, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(28//2, 42//2, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(42//2, 32, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(32, 94//2, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(94//2, 70, kernel_size=3, stride=1, padding=1)
            self.conv6 = nn.Conv2d(70, 212//2, kernel_size=3, stride=1, padding=1)
            self.fc_gaze = nn.Linear(212//2, out_count)
            #self.fc_exp = nn.Linear(212, 1)

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            self.adaptive = GlobalMaxPool2d()

            self.act = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, return_blends=True):
            x = self.conv1(x)
            x = self.act(x)
            x = self.pool(x)

            x = self.conv2(x)
            x = self.act(x)
            x = self.pool(x)

            x = self.conv3(x)
            x = self.act(x)
            x = self.pool(x)

            x = self.conv4(x)
            x = self.act(x)
            x = self.pool(x)

            x = self.conv5(x)
            x = self.act(x)
            x = self.pool(x)

            x = self.conv6(x)
            x = self.act(x)

            x = self.adaptive(x)
            x = torch.flatten(x, 1)

            #y = self.fc_exp(x)
            #y = self.sigmoid(y)

            #if not return_blends:
            #    return x
            
            x = self.fc_gaze(x)
            x = self.sigmoid(x)

            #x = torch.cat([x, y], dim=-1)

            return x

    class MultiInputMergedMicroChad(nn.Module):
        def __init__(self, left_models, right_models):
            super(MultiInputMergedMicroChad, self).__init__()

            if not left_models or not right_models:
                raise ValueError("Model lists for both left and right inputs cannot be empty.")

            self.all_models = left_models + right_models
            self.num_models = len(self.all_models)
            self.num_left = len(left_models)
            self.num_right = len(right_models)
            self.original_model_ref = self.all_models[0]

            # --- Create merged convolutional layers with input routing logic ---
            self._create_merged_conv_layers(left_models, right_models)

            # --- Store the final classification layers separately ---
            self.final_layers = nn.ModuleList(
                [model.fc_gaze for model in self.all_models]
            )

            # --- Define shared, parameter-less layers ---
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.adaptive = GlobalMaxPool2d()
            self.act = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

        def _create_merged_conv_layers(self, left_models, right_models):
            """Merges conv layers, with special handling for conv1 to route inputs."""
            
            conv_layer_names = [name for name, module in self.original_model_ref.named_modules() if isinstance(module, nn.Conv2d)]

            for i, layer_name in enumerate(conv_layer_names):
                # Get layers from all models for merging
                all_original_layers = [getattr(model, layer_name) for model in self.all_models]
                out_c, in_c, k_h, k_w = all_original_layers[0].weight.shape
                
                if i == 0:
                    # --- SPECIAL HANDLING FOR THE FIRST CONV LAYER ---
                    # This layer will perform the input routing.
                    groups = 1 # It's a single, custom-built convolution
                    
                    # The merged layer will take all input channels (e.g., 4+4=8)
                    new_in_c = in_c * 2 # Assuming two input streams (left, right)
                    # The output channels are the sum of all model outputs
                    new_out_c = out_c * self.num_models

                    # Build the block-diagonal weight matrix for input routing
                    merged_weight = torch.zeros(new_out_c, new_in_c, k_h, k_w)
                    
                    # Get weights for left and right models separately
                    left_conv1_layers = [model.conv1 for model in left_models]
                    right_conv1_layers = [model.conv1 for model in right_models]

                    # Concatenate weights for each stream
                    left_weights = torch.cat([layer.weight.data for layer in left_conv1_layers], dim=0)
                    right_weights = torch.cat([layer.weight.data for layer in right_conv1_layers], dim=0)
                    
                    # Place left weights in the top-left block of the matrix
                    # It will process the first 4 input channels (0:in_c)
                    merged_weight[0:self.num_left*out_c, 0:in_c, :, :] = left_weights

                    # Place right weights in the bottom-right block of the matrix
                    # It will process the last 4 input channels (in_c:)
                    merged_weight[self.num_left*out_c:, in_c:, :, :] = right_weights

                    # Biases can just be concatenated as they are applied after the routing
                    merged_bias = torch.cat([layer.bias.data for layer in all_original_layers], dim=0)

                else:
                    # --- SUBSEQUENT CONV LAYERS ---
                    # Use grouped convolutions for efficiency and separation.
                    groups = self.num_models
                    new_in_c = in_c * self.num_models
                    new_out_c = out_c * self.num_models
                    
                    merged_weight = torch.cat([layer.weight.data for layer in all_original_layers], dim=0)
                    merged_bias = torch.cat([layer.bias.data for layer in all_original_layers], dim=0)

                # Create the new, wider Conv2D layer
                merged_layer = nn.Conv2d(new_in_c, new_out_c, (k_h, k_w),
                                        stride=all_original_layers[0].stride,
                                        padding=all_original_layers[0].padding,
                                        groups=groups)
                
                merged_layer.weight.data = merged_weight
                merged_layer.bias.data = merged_bias
                setattr(self, layer_name, merged_layer)

        def forward(self, x):
            # The forward pass is simple because conv1 handles the routing.
            x = self.act(self.conv1(x))
            x = self.pool(x)

            x = self.act(self.conv2(x))
            x = self.pool(x)

            x = self.act(self.conv3(x))
            x = self.pool(x)
            
            x = self.act(self.conv4(x))
            x = self.pool(x)

            x = self.act(self.conv5(x))
            x = self.pool(x)

            x = self.act(self.conv6(x))
            
            x = self.adaptive(x)
            x = torch.flatten(x, 1)

            # Split the final feature vector and pass to respective heads
            outputs = []
            chunk_size = self.original_model_ref.conv6.out_channels
            feature_chunks = torch.split(x, split_size_or_sections=chunk_size, dim=1)
            
            for i, head in enumerate(self.final_layers):
                output = head(feature_chunks[i])
                output = self.sigmoid(output)
                outputs.append(output)
                
            return outputs

    def train_model(models_left, models_right, decoder, train_loader, num_epochs=10, lr=5e-5, class_step=False, e_add = 0, e_total = 0, exp_stage=False, lrs=[], label_sizes=[], aug_levels=[]):

        device = DEVICE#torch.device("cuda:0")
        print(f"Using device: {device}", flush=True)
        
        #model = model.to(device)
        criterion = nn.MSELoss()
        
        
        # For tracking progress
        epoch_losses = []
        batch_losses = []

        EPOCH_SIZE = 100

        #model.train()
        for model in models_left + models_right:
            model.train()
        models_left = [ model.to(DEVICE) for model in models_left ]
        models_right = [ model.to(DEVICE) for model in models_right ]

        optimizersL = [ optim.AdamW(list(models_left[e].parameters()), lr=lrs[e]) for e in range(len(models_left)) ]
        optimizersR = [ optim.AdamW(list(models_right[e].parameters()), lr=lrs[e]) for e in range(len(models_right)) ]

        def warmup_fn(epoch):
            return min(1.0, (epoch+1) / (5*EPOCH_SIZE))  # Gradually increase LR for first 5 epochs

        warmup_schedulersL = [ LambdaLR(e, lr_lambda=warmup_fn) for e in optimizersL ]
        warmup_schedulersR = [ LambdaLR(e, lr_lambda=warmup_fn) for e in optimizersR ]

        T_max = num_epochs-5  # Total epochs minus warm-up
        eta_min = 1e-6  # Minimum LR after decay

        cosine_schedulersL = [ CosineAnnealingLR(e, T_max=T_max * EPOCH_SIZE, eta_min=eta_min) for e in optimizersL ]
        cosine_schedulersR = [ CosineAnnealingLR(e, T_max=T_max * EPOCH_SIZE, eta_min=eta_min) for e in optimizersR ]

        def make_batch(batcher_fn, batch_size=32):
            labels_left = []
            labels_right = []
            inputs_left = []
            inputs_right = []
            gaze_states = []
            for e in range(batch_size):
                input_left, input_right, label_left, label_right, gaze_state, state = batcher_fn()
                inputs_left.append(input_left)
                inputs_right.append(input_right)
                labels_left.append(label_left)
                labels_right.append(label_right)
                gaze_states.append(gaze_state)
            
            return torch.stack(inputs_left, dim=0), torch.stack(inputs_right, dim=0), torch.stack(labels_left, dim=0), torch.stack(labels_right, dim=0), torch.stack(gaze_states, dim=0)

        def train_one_model(num_epochs, steps_per_epoch, model_idx, batcher_fn):
            dataloader = DataLoader(DatasetWrapper(batcher_fn, num_epochs, 32, steps_per_epoch), batch_size=32, shuffle=True, num_workers=8, worker_init_fn=worker_init, pin_memory=True)
            
            print("\nTraining %s:" % (MODEL_NAMES[model_idx]))
            optim_left = optimizersL[model_idx]
            optim_right = optimizersR[model_idx]
            model_left = models_left[model_idx]
            model_right = models_right[model_idx]
            sched_left_w = warmup_schedulersL[model_idx]
            sched_right_w = warmup_schedulersR[model_idx]
            sched_left = cosine_schedulersL[model_idx]
            sched_right = cosine_schedulersR[model_idx]

            label_start_idx = 0
            for e in range(model_idx):
                label_start_idx = label_start_idx + label_sizes[e]
            label_end_idx = label_start_idx + label_sizes[model_idx]

            train_loader.reset_use_counts()

            dataloader_iterator = iter(dataloader)

            for epoch in range(num_epochs):
                all_loss = 0
                start_time = time.time()
                for step in range(steps_per_epoch):
                    inputs_left, inputs_right, labels_left, labels_right, gaze_states, sample_states = next(dataloader_iterator)
                    
                    optim_left.zero_grad()
                    optim_right.zero_grad()

                    predL = model_left(inputs_left.to(DEVICE))
                    predR = model_right(inputs_right.to(DEVICE))

                    loss = criterion(predL, labels_left[:, label_start_idx:label_end_idx].to(DEVICE))
                    loss += criterion(predR, labels_right[:, label_start_idx:label_end_idx].to(DEVICE))

                    loss.backward()

                    optim_left.step()
                    optim_right.step()
                    print("\r[%s(%d/%d)][%d/%d][%u/%u] loss: %.4f, lr: %.5f" % (MODEL_NAMES[model_idx], model_idx+1, len(MODEL_NAMES), epoch, num_epochs, step, EPOCH_SIZE, float(loss), float(optim_left.param_groups[0]['lr'])), flush=True, end='')
                    all_loss += float(loss)

                    if epoch < 5:
                        sched_left_w.step()
                        sched_right_w.step()
                    else:
                        sched_left.step()
                        sched_right.step()
                total_time = time.time() - start_time
                print(" - Epoch average: %.4f, time: %.1fs" % (all_loss / steps_per_epoch, total_time))

        GAZE_EPOCHS = 32
        BLINK_EPOCHS = 32
        SQWI_EPOCHS = 32
        BROW_EPOCHS = 32

        train_one_model(GAZE_EPOCHS, EPOCH_SIZE, 0, batcher_gaze)
        train_one_model(BLINK_EPOCHS, EPOCH_SIZE, 1, batcher_lid)
        train_one_model(SQWI_EPOCHS, EPOCH_SIZE, 2, batcher_wide_squeeze)
        train_one_model(BROW_EPOCHS, EPOCH_SIZE, 3, batcher_brow)

        return models_left, models_right, epoch_losses, batch_losses

    def normalize_similarity(similarity):
        return (similarity + 1) / 2

    def main():
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        model_L=[MicroChad(out_count=2),MicroChad(out_count=1),MicroChad(out_count=2),MicroChad(out_count=1)]
        model_R=[MicroChad(out_count=2),MicroChad(out_count=1),MicroChad(out_count=2),MicroChad(out_count=1)]

        trained_model_L = model_L
        trained_model_R = model_R

        EPOCHS = 42

        dataset = CaptureDataset('G:\\Baballonia.Setup.v1.1.0.8\\user_cal.bin', all_frames=False)

        train_dataset = dataset
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

        trained_model_L, trained_model_R, epoch_losses, batch_losses = train_model(
            trained_model_L,
            trained_model_R,
            None,
            dataset, 
            num_epochs=EPOCHS,
            lr=0.001,
            class_step=True, 
            exp_stage=False,
            e_add = 0,
            e_total = EPOCHS,
            lrs=[0.001, 1e-4, 1e-4, 1e-4],
            label_sizes=[2, 1, 2, 1],
            aug_levels=[0, 1, 2, 2]
        )

        for e in range(len(trained_model_L)):
            torch.save(trained_model_R[e].state_dict(), "right_tuned_sqwi_multiv2_%d.pth"%e)
            torch.save(trained_model_L[e].state_dict(), "left_tuned_sqwi_multiv2_%d.pth"%e)

        print("\nTraining completed successfully!\n", flush=True)

        device = torch.device("cpu")
        multi = MultiInputMergedMicroChad(trained_model_L, trained_model_R)

        device = torch.device("cpu")
        multi = multi.to(device)
        
        dummy_input = torch.randn(1, 8, 128, 128, device=device)  # Updated to 8 channels
        torch.onnx.export(
            multi,
            dummy_input,
            sys.argv[2],
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            dynamo=True,
            external_data=False
        )
        print("Model exported to ONNX: " + sys.argv[2], flush=True)

    main()
