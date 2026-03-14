"""
vision_connector.py — MediaPipe vision connector for wavehaze

Captures webcam via OpenCV, runs MediaPipe hand/pose/segmentation,
writes results to WaveHaze_Vision shared memory.

Requirements:
    pip install mediapipe opencv-python numpy

Usage:
    python vision_connector.py                  # all features
    python vision_connector.py --hands          # hands only
    python vision_connector.py --pose           # pose only
    python vision_connector.py --segment        # segmentation only
    python vision_connector.py --camera 1       # use camera index 1

wavehaze reads the shared memory automatically when present.
"""

import argparse
import ctypes
import ctypes.wintypes
import math
import struct
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

# ── Shared memory constants (must match vision.h) ──────────

VISION_MAGIC = 0x56495332  # "VIS2"
VISION_VERSION = 1
VISION_SHM_NAME = "WaveHaze_Vision"
VISION_SHM_SIZE = 512 * 1024

HEADER_OFFSET = 0x0000
HANDS_OFFSET  = 0x0100
POSE_OFFSET   = 0x0300
FACE_OFFSET   = 0x0510
MASK_OFFSET   = 0x1B00

HAND_POINTS = 21
POSE_POINTS = 33
MASK_W = 640
MASK_H = 480

# Feature flags
HAS_HANDS = 1 << 0
HAS_POSE  = 1 << 1
HAS_FACE  = 1 << 2
HAS_MASK  = 1 << 3

# ── Windows shared memory ──────────────────────────────────

kernel32 = ctypes.windll.kernel32

PAGE_READWRITE = 0x04
FILE_MAP_ALL_ACCESS = 0xF001F
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def create_shared_memory():
    """Create the WaveHaze_Vision shared memory region."""
    handle = kernel32.CreateFileMappingA(
        INVALID_HANDLE_VALUE, None, PAGE_READWRITE,
        0, VISION_SHM_SIZE, VISION_SHM_NAME.encode()
    )
    if not handle:
        raise RuntimeError(f"CreateFileMapping failed: {ctypes.GetLastError()}")

    base = kernel32.MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, VISION_SHM_SIZE)
    if not base:
        kernel32.CloseHandle(handle)
        raise RuntimeError(f"MapViewOfFile failed: {ctypes.GetLastError()}")

    # Zero it out
    ctypes.memset(base, 0, VISION_SHM_SIZE)
    return handle, base


def write_header(base, features, source_name="MediaPipe"):
    """Write VisionHeader at offset 0."""
    name_bytes = source_name.encode()[:31].ljust(32, b'\x00')
    ver_bytes = b"1.0\x00".ljust(16, b'\x00')

    header = struct.pack(
        '<IIIIqQ32s16s',
        VISION_MAGIC,
        VISION_VERSION,
        features,
        0,  # flags
        0,  # sequence (will be updated)
        0,  # timestamp_us
        name_bytes,
        ver_bytes,
    )
    ctypes.memmove(base + HEADER_OFFSET, header, len(header))


def update_sequence(base, seq, hand_count=0, pose_detected=0,
                    gesture_l=0, gesture_r=0,
                    hand_spread_l=0, hand_spread_r=0,
                    body_cx=0.5, body_cy=0.5, body_h=0):
    """Update dynamic header fields."""
    # sequence at offset 16 (after magic, version, features, flags)
    struct.pack_into('<q', ctypes.string_at(base, VISION_SHM_SIZE), 16, seq)
    # Actually, use memmove for the volatile fields
    data = struct.pack(
        '<qQxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxIIIIIIIfffff',
        seq,
        int(time.monotonic() * 1_000_000),
        # skip source_name + source_version (48 bytes, already written)
        hand_count, pose_detected, 0,  # face_detected
        MASK_W, MASK_H,
        gesture_l, gesture_r,
        hand_spread_l, hand_spread_r,
        body_h, body_cx, body_cy,
    )
    # This is getting complex, let's just write fields individually
    off = HEADER_OFFSET + 16  # after magic(4) + version(4) + features(4) + flags(4)
    ctypes.memmove(base + off, struct.pack('<q', seq), 8)
    off += 8
    ctypes.memmove(base + off, struct.pack('<Q', int(time.monotonic() * 1_000_000)), 8)

    # Skip source_name(32) + source_version(16) = 48 bytes
    off = HEADER_OFFSET + 80  # 16 + 8 + 8 + 32 + 16
    counts = struct.pack(
        '<IIIIIIIfffff',
        hand_count, pose_detected, 0,
        MASK_W, MASK_H,
        gesture_l, gesture_r,
        hand_spread_l, hand_spread_r,
        body_h, body_cx, body_cy,
    )
    ctypes.memmove(base + off, counts, len(counts))


def write_hand_landmarks(base, hand_idx, landmarks):
    """Write 21 hand landmarks (x, y, confidence) for hand 0 or 1."""
    off = HANDS_OFFSET + hand_idx * HAND_POINTS * 12  # 3 floats × 4 bytes
    for i, lm in enumerate(landmarks[:HAND_POINTS]):
        data = struct.pack('<fff', lm.x, lm.y, lm.visibility if hasattr(lm, 'visibility') else 1.0)
        ctypes.memmove(base + off + i * 12, data, 12)


def write_pose_landmarks(base, landmarks):
    """Write 33 pose landmarks (x, y, z, confidence)."""
    for i, lm in enumerate(landmarks[:POSE_POINTS]):
        data = struct.pack('<ffff', lm.x, lm.y, lm.z, lm.visibility)
        ctypes.memmove(base + POSE_OFFSET + i * 16, data, 16)


def write_segmentation_mask(base, mask):
    """Write segmentation mask (uint8, 0-255)."""
    # Resize to standard dimensions
    if mask.shape[:2] != (MASK_H, MASK_W):
        mask = cv2.resize(mask, (MASK_W, MASK_H))

    # Convert to uint8 if float
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask = (mask * 255).astype(np.uint8)

    data = mask.tobytes()
    ctypes.memmove(base + MASK_OFFSET, data, min(len(data), MASK_W * MASK_H))


def compute_hand_spread(landmarks):
    """Compute how 'open' a hand is (0=fist, 1=fully open)."""
    if not landmarks or len(landmarks) < 21:
        return 0.0

    # Compare fingertip distances to wrist
    wrist = landmarks[0]
    tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]  # thumb, index, middle, ring, pinky

    dists = []
    for tip in tips:
        d = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        dists.append(d)

    avg_dist = sum(dists) / len(dists)
    return min(avg_dist * 4.0, 1.0)  # normalize roughly to 0..1


def detect_gesture(landmarks):
    """Simple gesture detection. Returns enum value."""
    if not landmarks or len(landmarks) < 21:
        return 0  # none

    # Check if fingers are extended
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    extended = []
    for tip, pip in zip(tips, pips):
        extended.append(landmarks[tip].y < landmarks[pip].y)

    n_extended = sum(extended)

    if n_extended >= 4:
        return 1  # open hand
    elif n_extended == 0:
        return 2  # closed fist
    elif extended[1] and not extended[2] and not extended[3] and not extended[4]:
        return 4  # point (index only)
    elif extended[1] and extended[2] and not extended[3] and not extended[4]:
        return 5  # peace
    elif n_extended <= 2:
        # Check pinch (thumb tip close to index tip)
        dx = landmarks[4].x - landmarks[8].x
        dy = landmarks[4].y - landmarks[8].y
        if math.sqrt(dx*dx + dy*dy) < 0.05:
            return 3  # pinch

    return 0  # none


def main():
    parser = argparse.ArgumentParser(description="MediaPipe vision connector for wavehaze")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--hands", action="store_true", help="Enable hand tracking")
    parser.add_argument("--pose", action="store_true", help="Enable pose estimation")
    parser.add_argument("--segment", action="store_true", help="Enable segmentation")
    parser.add_argument("--all", action="store_true", help="Enable all features")
    parser.add_argument("--show", action="store_true", help="Show debug preview window")
    args = parser.parse_args()

    # Default to all if nothing specified
    if not (args.hands or args.pose or args.segment):
        args.all = True

    use_hands = args.hands or args.all
    use_pose = args.pose or args.all
    use_segment = args.segment or args.all

    features = 0
    if use_hands:   features |= HAS_HANDS
    if use_pose:    features |= HAS_POSE
    if use_segment: features |= HAS_MASK

    print(f"Vision connector starting...")
    print(f"  Camera: {args.camera}")
    print(f"  Hands: {use_hands}, Pose: {use_pose}, Segment: {use_segment}")

    # Create shared memory
    handle, base = create_shared_memory()
    write_header(base, features)
    print(f"  Shared memory: {VISION_SHM_NAME} ({VISION_SHM_SIZE // 1024} KB)")

    # Init MediaPipe
    mp_hands = None
    mp_pose = None
    mp_selfie = None

    if use_hands:
        mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    if use_pose:
        mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=use_segment,
        )

    if use_segment and not use_pose:
        mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"  Running (Ctrl+C to stop)...")

    seq = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_count = 0
            pose_detected = 0
            gesture_l = 0
            gesture_r = 0
            hand_spread_l = 0.0
            hand_spread_r = 0.0
            body_cx = 0.5
            body_cy = 0.5
            body_h = 0.0

            # Hand tracking
            if mp_hands:
                results = mp_hands.process(rgb)
                if results.multi_hand_landmarks:
                    hand_count = len(results.multi_hand_landmarks)
                    for i, hand_lms in enumerate(results.multi_hand_landmarks[:2]):
                        # Determine which hand (left/right)
                        handedness = results.multi_handedness[i].classification[0]
                        # MediaPipe mirrors: "Left" in image = right hand
                        hand_idx = 1 if handedness.label == "Left" else 0

                        write_hand_landmarks(base, hand_idx, hand_lms.landmark)

                        spread = compute_hand_spread(hand_lms.landmark)
                        gesture = detect_gesture(hand_lms.landmark)

                        if hand_idx == 0:
                            hand_spread_l = spread
                            gesture_l = gesture
                        else:
                            hand_spread_r = spread
                            gesture_r = gesture

            # Pose estimation
            if mp_pose:
                results = mp_pose.process(rgb)
                if results.pose_landmarks:
                    pose_detected = 1
                    write_pose_landmarks(base, results.pose_landmarks.landmark)

                    # Body center and height
                    lms = results.pose_landmarks.landmark
                    xs = [lm.x for lm in lms if lm.visibility > 0.5]
                    ys = [lm.y for lm in lms if lm.visibility > 0.5]
                    if xs and ys:
                        body_cx = sum(xs) / len(xs)
                        body_cy = sum(ys) / len(ys)
                        body_h = max(ys) - min(ys)

                # Segmentation from pose
                if use_segment and results.segmentation_mask is not None:
                    write_segmentation_mask(base, results.segmentation_mask)

            # Standalone segmentation
            if mp_selfie:
                results = mp_selfie.process(rgb)
                if results.segmentation_mask is not None:
                    write_segmentation_mask(base, results.segmentation_mask)

            # Update header
            seq += 1
            update_sequence(
                base, seq,
                hand_count=hand_count,
                pose_detected=pose_detected,
                gesture_l=gesture_l,
                gesture_r=gesture_r,
                hand_spread_l=hand_spread_l,
                hand_spread_r=hand_spread_r,
                body_cx=body_cx,
                body_cy=body_cy,
                body_h=body_h,
            )

            # Debug preview
            if args.show:
                cv2.imshow("Vision Connector", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

    except KeyboardInterrupt:
        print("\nStopping...")

    # Cleanup
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    if mp_hands: mp_hands.close()
    if mp_pose:  mp_pose.close()
    if mp_selfie: mp_selfie.close()

    # Zero magic to signal shutdown
    ctypes.memmove(base, struct.pack('<I', 0), 4)
    kernel32.UnmapViewOfFile(base)
    kernel32.CloseHandle(handle)

    print("Vision connector stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
