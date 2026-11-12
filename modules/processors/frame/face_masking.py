import cv2
import numpy as np
from modules.typing import Face, Frame
import modules.globals
from modules.gpu_processing import gpu_gaussian_blur, gpu_resize

def apply_color_transfer(source, target):
    """
    Apply color transfer from target to source image using LAB color space.
    Routes through the unified, high-performance clamped Lab transfer.
    """
    from modules.processors.frame.face_swapper import apply_color_transfer as _swapper_act
    return _swapper_act(source, target)

def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Convert landmarks to int32
        landmarks = landmarks.astype(np.int32)

        # Extract facial features
        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]
        right_eye = landmarks[33:42]
        right_eye_brow = landmarks[43:51]
        left_eye = landmarks[87:96]
        left_eye_brow = landmarks[97:105]

        # Calculate padding
        padding = int(
            np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05
        )  # 5% of face width

        # Create a slightly larger convex hull for padding
        face_outline = landmarks[0:33]
        hull = cv2.convexHull(face_outline)
        # Vectorized hull padding — expand each point outward from center
        center = np.mean(face_outline, axis=0, dtype=np.float32)
        hull_pts = hull.reshape(-1, 2).astype(np.float32)
        directions = hull_pts - center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)  # avoid division by zero
        directions /= norms
        hull_padded = (hull_pts + directions * padding).astype(np.int32)

        # Fill the padded convex hull
        cv2.fillConvexPoly(mask, hull_padded, 255)

        # Smooth the mask edges (GPU-accelerated when available)
        mask = gpu_gaussian_blur(mask, (5, 5), 3)

    return mask

def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    lower_lip_polygon = None
    mouth_box = (0,0,0,0)

    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Use outer mouth landmarks (52-71) to capture the full mouth area
        lower_lip_order = list(range(52, 72))
        
        if max(lower_lip_order) >= landmarks.shape[0]:
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)

        # Calculate the center of the landmarks
        center = np.mean(lower_lip_landmarks, axis=0)

        # Expand the landmarks outward using the mouth_mask_size
        mouth_mask_size = getattr(modules.globals, "mouth_mask_size", 0.0) # 0-100 slider
        expansion_factor = 1 + (mouth_mask_size / 100.0) * 2.5

        # Expand with extra downward bias toward chin
        offsets = lower_lip_landmarks - center
        chin_bias = 1 + (mouth_mask_size / 100.0) * 1.5
        scale_y = np.where(offsets[:, 1] > 0, expansion_factor * chin_bias, expansion_factor)
        expanded_landmarks = lower_lip_landmarks.copy()
        expanded_landmarks[:, 0] = center[0] + offsets[:, 0] * expansion_factor
        expanded_landmarks[:, 1] = center[1] + offsets[:, 1] * scale_y

        # Convert back to integer coordinates
        expanded_landmarks = expanded_landmarks.astype(np.int32)

        # Calculate bounding box for the expanded lower mouth
        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        # Add some padding to the bounding box
        padding = int((max_x - min_x) * 0.1)  # 10% padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)

        # Ensure the bounding box dimensions are valid
        if max_x <= min_x or max_y <= min_y:
            if (max_x - min_x) <= 1:
                max_x = min_x + 1
            if (max_y - min_y) <= 1:
                max_y = min_y + 1

        # Create the mask
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        # Shift polygon coordinates relative to the ROI's top-left corner
        polygon_relative_to_roi = expanded_landmarks - [min_x, min_y]
        cv2.fillPoly(mask_roi, [polygon_relative_to_roi], 255)

        # Apply Gaussian blur to soften the mask edges (GPU-accelerated when available)
        mask_roi = gpu_gaussian_blur(mask_roi, (15, 15), 5)

        # Place the mask ROI in the full-sized mask
        mask[min_y:max_y, min_x:max_x] = mask_roi

        # Extract the masked area from the frame
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()

        # Return the expanded lower lip polygon in original frame coordinates
        lower_lip_polygon = expanded_landmarks
        mouth_box = (min_x, min_y, max_x, max_y)

    return mask, mouth_cutout, mouth_box, lower_lip_polygon

def create_eyes_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    eyes_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Left eye landmarks (87-96) and right eye landmarks (33-42)
        left_eye = landmarks[87:96]
        right_eye = landmarks[33:42]
        
        # Calculate centers and dimensions for each eye
        left_eye_center = np.mean(left_eye, axis=0).astype(np.int32)
        right_eye_center = np.mean(right_eye, axis=0).astype(np.int32)
        
        def get_eye_dimensions(eye_points):
            x_coords = eye_points[:, 0]
            y_coords = eye_points[:, 1]
            down_size = getattr(modules.globals, "mask_down_size", 0.1)
            eye_size = getattr(modules.globals, "eyes_mask_size", 0.0)
            width = int((np.max(x_coords) - np.min(x_coords)) * (1 + down_size * eye_size))
            height = int((np.max(y_coords) - np.min(y_coords)) * (1 + down_size * eye_size))
            return width, height
        
        left_width, left_height = get_eye_dimensions(left_eye)
        right_width, right_height = get_eye_dimensions(right_eye)
        
        # Add extra padding
        padding = int(max(left_width, right_width) * 0.2)
        
        # Calculate bounding box for both eyes
        min_x = min(left_eye_center[0] - left_width//2, right_eye_center[0] - right_width//2) - padding
        max_x = max(left_eye_center[0] + left_width//2, right_eye_center[0] + right_width//2) + padding
        min_y = min(left_eye_center[1] - left_height//2, right_eye_center[1] - right_height//2) - padding
        max_y = max(left_eye_center[1] + left_height//2, right_eye_center[1] + right_height//2) + padding
        
        # Ensure coordinates are within frame bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(frame.shape[1], max_x)
        max_y = min(frame.shape[0], max_y)
        
        # Create mask for the eyes region
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        
        # Draw ellipses for both eyes
        left_center = (left_eye_center[0] - min_x, left_eye_center[1] - min_y)
        right_center = (right_eye_center[0] - min_x, right_eye_center[1] - min_y)
        
        # Calculate axes lengths (half of width and height)
        left_axes = (left_width//2, left_height//2)
        right_axes = (right_width//2, right_height//2)
        
        # Draw filled ellipses
        cv2.ellipse(mask_roi, left_center, left_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(mask_roi, right_center, right_axes, 0, 0, 360, 255, -1)
        
        # Apply Gaussian blur to soften mask edges (GPU-accelerated when available)
        mask_roi = gpu_gaussian_blur(mask_roi, (15, 15), 5)
        
        # Place the mask ROI in the full-sized mask
        mask[min_y:max_y, min_x:max_x] = mask_roi
        
        # Extract the masked area from the frame
        eyes_cutout = frame[min_y:max_y, min_x:max_x].copy()
        
        # Create polygon points for visualization
        def create_ellipse_points(center, axes):
            t = np.linspace(0, 2*np.pi, 32)
            x = center[0] + axes[0] * np.cos(t)
            y = center[1] + axes[1] * np.sin(t)
            return np.column_stack((x, y)).astype(np.int32)
        
        # Generate points for both ellipses
        left_points = create_ellipse_points((left_eye_center[0], left_eye_center[1]), (left_width//2, left_height//2))
        right_points = create_ellipse_points((right_eye_center[0], right_eye_center[1]), (right_width//2, right_height//2))
        
        # Combine points for both eyes
        eyes_polygon = np.vstack([left_points, right_points])
        
    return mask, eyes_cutout, (min_x, min_y, max_x, max_y), eyes_polygon

def create_curved_eyebrow(points):
    if len(points) >= 5:
        # Sort points by x-coordinate
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]
        
        # Calculate dimensions
        x_min, y_min = np.min(sorted_points, axis=0)
        x_max, y_max = np.max(sorted_points, axis=0)
        width = x_max - x_min
        height = y_max - y_min
        
        # Create more points for smoother curve
        num_points = 50
        x = np.linspace(x_min, x_max, num_points)
        
        # Fit quadratic curve through points for more natural arch
        coeffs = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], 2)
        y = np.polyval(coeffs, x)
        
        # Increased offsets to create more separation
        top_offset = height * 0.5  # Increased from 0.3 to shift up more
        bottom_offset = height * 0.2  # Increased from 0.1 to shift down more
        
        # Create smooth curves
        top_curve = y - top_offset
        bottom_curve = y + bottom_offset
        
        # Create curved endpoints with more pronounced taper
        end_points = 5
        start_x = np.linspace(x[0] - width * 0.15, x[0], end_points)  # Increased taper
        end_x = np.linspace(x[-1], x[-1] + width * 0.15, end_points)  # Increased taper
        
        # Create tapered ends
        start_curve = np.column_stack((
            start_x,
            np.linspace(bottom_curve[0], top_curve[0], end_points)
        ))
        end_curve = np.column_stack((
            end_x,
            np.linspace(bottom_curve[-1], top_curve[-1], end_points)
        ))
        
        # Combine all points to form a smooth contour
        contour_points = np.vstack([
            start_curve,
            np.column_stack((x, top_curve)),
            end_curve,
            np.column_stack((x[::-1], bottom_curve[::-1]))
        ])
        
        # Add slight padding for better coverage
        center = np.mean(contour_points, axis=0)
        vectors = contour_points - center
        padded_points = center + vectors * 1.2  # Increased padding slightly
        
        return padded_points
    return points

def create_eyebrows_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    eyebrows_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Left eyebrow landmarks (97-105) and right eyebrow landmarks (43-51)
        left_eyebrow = landmarks[97:105].astype(np.float32)
        right_eyebrow = landmarks[43:51].astype(np.float32)
        
        # Calculate centers and dimensions for each eyebrow
        left_center = np.mean(left_eyebrow, axis=0)
        right_center = np.mean(right_eyebrow, axis=0)
        
        # Calculate bounding box with padding adjusted by size
        all_points = np.vstack([left_eyebrow, right_eyebrow])
        padding_factor = modules.globals.eyebrows_mask_size
        min_x = np.min(all_points[:, 0]) - 25 * padding_factor
        max_x = np.max(all_points[:, 0]) + 25 * padding_factor
        min_y = np.min(all_points[:, 1]) - 20 * padding_factor
        max_y = np.max(all_points[:, 1]) + 15 * padding_factor
        
        # Ensure coordinates are within frame bounds
        min_x = max(0, int(min_x))
        min_y = max(0, int(min_y))
        max_x = min(frame.shape[1], int(max_x))
        max_y = min(frame.shape[0], int(max_y))
        
        # Create mask for the eyebrows region
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        
        try:
            # Convert points to local coordinates
            left_local = left_eyebrow - [min_x, min_y]
            right_local = right_eyebrow - [min_x, min_y]
            
            def create_curved_eyebrow(points):
                if len(points) >= 5:
                    # Sort points by x-coordinate
                    sorted_idx = np.argsort(points[:, 0])
                    sorted_points = points[sorted_idx]
                    
                    # Calculate dimensions
                    x_min, y_min = np.min(sorted_points, axis=0)
                    x_max, y_max = np.max(sorted_points, axis=0)
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Create more points for smoother curve
                    num_points = 50
                    x = np.linspace(x_min, x_max, num_points)
                    
                    # Fit quadratic curve through points for more natural arch
                    coeffs = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], 2)
                    y = np.polyval(coeffs, x)
                    
                    # Increased offsets to create more separation
                    top_offset = height * 0.5  # Increased from 0.3 to shift up more
                    bottom_offset = height * 0.2  # Increased from 0.1 to shift down more
                    
                    # Create smooth curves
                    top_curve = y - top_offset
                    bottom_curve = y + bottom_offset
                    
                    # Create curved endpoints with more pronounced taper
                    end_points = 5
                    start_x = np.linspace(x[0] - width * 0.15, x[0], end_points)  # Increased taper
                    end_x = np.linspace(x[-1], x[-1] + width * 0.15, end_points)  # Increased taper
                    
                    # Create tapered ends
                    start_curve = np.column_stack((
                        start_x,
                        np.linspace(bottom_curve[0], top_curve[0], end_points)
                    ))
                    end_curve = np.column_stack((
                        end_x,
                        np.linspace(bottom_curve[-1], top_curve[-1], end_points)
                    ))
                    
                    # Combine all points to form a smooth contour
                    contour_points = np.vstack([
                        start_curve,
                        np.column_stack((x, top_curve)),
                        end_curve,
                        np.column_stack((x[::-1], bottom_curve[::-1]))
                    ])
                    
                    # Add slight padding for better coverage
                    center = np.mean(contour_points, axis=0)
                    vectors = contour_points - center
                    padded_points = center + vectors * 1.2  # Increased padding slightly
                    
                    return padded_points
                return points
            
            # Generate and draw eyebrow shapes
            left_shape = create_curved_eyebrow(left_local)
            right_shape = create_curved_eyebrow(right_local)
            
            # Apply multi-stage blurring for natural feathering (GPU-accelerated when available)
            # First, strong Gaussian blur for initial softening
            mask_roi = gpu_gaussian_blur(mask_roi, (21, 21), 7)
            
            # Second, medium blur for transition areas
            mask_roi = gpu_gaussian_blur(mask_roi, (11, 11), 3)
            
            # Finally, light blur for fine details
            mask_roi = gpu_gaussian_blur(mask_roi, (5, 5), 1)
            
            # Normalize mask values
            mask_roi = cv2.normalize(mask_roi, None, 0, 255, cv2.NORM_MINMAX)
            
            # Place the mask ROI in the full-sized mask
            mask[min_y:max_y, min_x:max_x] = mask_roi
            
            # Extract the masked area from the frame
            eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()
            
            # Combine points for visualization
            eyebrows_polygon = np.vstack([
                left_shape + [min_x, min_y],
                right_shape + [min_x, min_y]
            ]).astype(np.int32)
            
        except Exception as e:
            # Fallback to simple polygons if curve fitting fails
            left_local = left_eyebrow - [min_x, min_y]
            right_local = right_eyebrow - [min_x, min_y]
            cv2.fillPoly(mask_roi, [left_local.astype(np.int32)], 255)
            cv2.fillPoly(mask_roi, [right_local.astype(np.int32)], 255)
            mask_roi = gpu_gaussian_blur(mask_roi, (21, 21), 7)
            mask[min_y:max_y, min_x:max_x] = mask_roi
            eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()
            eyebrows_polygon = np.vstack([left_eyebrow, right_eyebrow]).astype(np.int32)
        
    return mask, eyebrows_cutout, (min_x, min_y, max_x, max_y), eyebrows_polygon

def apply_mask_area(
    frame: np.ndarray,
    cutout: np.ndarray,
    box: tuple,
    face_mask: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        cutout is None
        or box_width is None
        or box_height is None
        or face_mask is None
        or polygon is None
    ):
        return frame

    try:
        resized_cutout = gpu_resize(cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_cutout.shape:
            resized_cutout = gpu_resize(
                resized_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_area = apply_color_transfer(resized_cutout, roi)

        # Create mask for the area
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        
        # Split points for left and right parts if needed
        if len(polygon) > 50:  # Arbitrary threshold to detect if we have multiple parts
            mid_point = len(polygon) // 2
            left_points = polygon[:mid_point] - [min_x, min_y]
            right_points = polygon[mid_point:] - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [left_points], 255)
            cv2.fillPoly(polygon_mask, [right_points], 255)
        else:
            adjusted_polygon = polygon - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        # Apply strong initial feathering (GPU-accelerated when available)
        polygon_mask = gpu_gaussian_blur(polygon_mask, (21, 21), 7)

        # Apply additional feathering
        feather_amount = min(
            30,
            box_width // modules.globals.mask_feather_ratio,
            box_height // modules.globals.mask_feather_ratio,
        )
        feathered_mask = cv2.GaussianBlur(
            polygon_mask.astype(np.float32), (0, 0), feather_amount
        )
        max_val = feathered_mask.max()
        if max_val > 1e-6:
            feathered_mask *= np.float32(1.0 / max_val)

        # Apply additional smoothing to the mask edges
        feathered_mask = cv2.GaussianBlur(feathered_mask, (5, 5), 1)

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (face_mask_roi.astype(np.float32) * np.float32(1.0 / 255.0))

        combined_mask_3ch = combined_mask[:, :, np.newaxis]
        inv_mask = np.float32(1.0) - combined_mask_3ch
        blended = (
            color_corrected_area * combined_mask_3ch + roi * inv_mask
        ).astype(np.uint8)

        # Apply face mask to blended result
        face_mask_f32 = face_mask_roi[:, :, np.newaxis].astype(np.float32) * np.float32(1.0 / 255.0)
        face_mask_3channel = np.broadcast_to(face_mask_f32, blended.shape)
        final_blend = blended * face_mask_3channel + roi * (np.float32(1.0) - face_mask_3channel)

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception as e:
        pass

    return frame

def draw_mask_visualization(
    frame: Frame,
    mask_data: tuple,
    label: str,
    draw_method: str = "polygon"
) -> Frame:
    mask, cutout, (min_x, min_y, max_x, max_y), polygon = mask_data

    vis_frame = frame.copy()

    # Ensure coordinates are within frame bounds
    height, width = vis_frame.shape[:2]
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(width, max_x), min(height, max_y)

    if draw_method == "ellipse" and len(polygon) > 50:  # For eyes
        # Split points for left and right parts
        mid_point = len(polygon) // 2
        left_points = polygon[:mid_point]
        right_points = polygon[mid_point:]
        
        try:
            # Fit ellipses to points - need at least 5 points
            if len(left_points) >= 5 and len(right_points) >= 5:
                # Convert points to the correct format for ellipse fitting
                left_points = left_points.astype(np.float32)
                right_points = right_points.astype(np.float32)
                
                # Fit ellipses
                left_ellipse = cv2.fitEllipse(left_points)
                right_ellipse = cv2.fitEllipse(right_points)
                
                # Draw the ellipses
                cv2.ellipse(vis_frame, left_ellipse, (0, 255, 0), 2)
                cv2.ellipse(vis_frame, right_ellipse, (0, 255, 0), 2)
        except Exception as e:
            # If ellipse fitting fails, draw simple rectangles as fallback
            left_rect = cv2.boundingRect(left_points)
            right_rect = cv2.boundingRect(right_points)
            cv2.rectangle(vis_frame, 
                        (left_rect[0], left_rect[1]), 
                        (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), 
                        (0, 255, 0), 2)
            cv2.rectangle(vis_frame,
                        (right_rect[0], right_rect[1]),
                        (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]),
                        (0, 255, 0), 2)
    else:  # For mouth and eyebrows
        # Draw the polygon
        if len(polygon) > 50:  # If we have multiple parts
            mid_point = len(polygon) // 2
            left_points = polygon[:mid_point]
            right_points = polygon[mid_point:]
            cv2.polylines(vis_frame, [left_points], True, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.polylines(vis_frame, [right_points], True, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.polylines(vis_frame, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)

    # Add label
    cv2.putText(
        vis_frame,
        label,
        (min_x, min_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return vis_frame


def create_glasses_mask(
    face: Face, frame: Frame
) -> tuple:
    """Create a feathered mask covering glasses frames, lenses, and bridge."""
    if face is None or frame is None:
        return np.zeros((1, 1), dtype=np.uint8), None, (0, 0, 0, 0), None

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    glasses_cutout = None
    glasses_polygon = None
    glasses_box = (0, 0, 0, 0)

    glasses_mask_size = getattr(modules.globals, "glasses_mask_size", 0.0)
    # Generous base expansion (1.22) covers standard glasses frames and temples by default;
    # glasses_mask_size allows further expansion up to 1.7x for oversized frames.
    expansion_factor = 1.22 + (glasses_mask_size / 100.0) * 0.48

    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is not None and isinstance(landmarks, np.ndarray) and landmarks.shape[0] >= 106:
        # Right eye: 33-42, right eyebrow: 43-51, left eye: 87-96, left eyebrow: 97-105, bridge: 72-75
        r_eye = landmarks[33:43].astype(np.float32)
        r_brow = landmarks[43:52].astype(np.float32)
        l_eye = landmarks[87:97].astype(np.float32)
        l_brow = landmarks[97:106].astype(np.float32)
        bridge = landmarks[72:76].astype(np.float32)

        r_center = np.mean(r_eye, axis=0)
        l_center = np.mean(l_eye, axis=0)
        r_eye_h = max(4.0, float(np.max(r_eye[:, 1]) - np.min(r_eye[:, 1])))
        l_eye_h = max(4.0, float(np.max(l_eye[:, 1]) - np.min(l_eye[:, 1])))

        # Points extending below eyes for lower frame rim (1.35x eye height covers full bottom rim & lenses)
        r_lower = r_eye + np.array([0, r_eye_h * 1.35], dtype=np.float32)
        l_lower = l_eye + np.array([0, l_eye_h * 1.35], dtype=np.float32)

        r_all = np.vstack([r_brow, r_eye, r_lower])
        l_all = np.vstack([l_brow, l_eye, l_lower])
        r_exp = r_center + (r_all - r_center) * expansion_factor
        l_exp = l_center + (l_all - l_center) * expansion_factor

        # Pad nose bridge laterally to cover frame bridge and nose pads
        eye_dist = max(10.0, float(np.linalg.norm(l_center - r_center)))
        bridge_w = max(4.0, eye_dist * 0.12)
        bridge_left = bridge - np.array([bridge_w, 0], dtype=np.float32)
        bridge_right = bridge + np.array([bridge_w, 0], dtype=np.float32)

        all_glasses_pts = np.vstack([r_exp, bridge, bridge_left, bridge_right, l_exp])
        hull = cv2.convexHull(all_glasses_pts.astype(np.float32)).reshape(-1, 2)
        glasses_polygon = hull.astype(np.int32)
    elif hasattr(face, "kps") and face.kps is not None and getattr(face.kps, "shape", None) == (5, 2):
        # 5-point fallback
        kps = face.kps.astype(np.float32)
        eye_dist = max(10.0, float(np.linalg.norm(kps[1] - kps[0])))
        rx = eye_dist * 0.42 * expansion_factor
        ry = eye_dist * 0.38 * expansion_factor
        t = np.linspace(0, 2 * np.pi, 24)
        c0 = kps[0]
        pts0 = np.column_stack([c0[0] + rx * np.cos(t), c0[1] + ry * np.sin(t)])
        c1 = kps[1]
        pts1 = np.column_stack([c1[0] + rx * np.cos(t), c1[1] + ry * np.sin(t)])
        bridge = np.array([
            (kps[0] + kps[1]) * 0.5 - [eye_dist * 0.1, ry * 0.2],
            (kps[0] + kps[1]) * 0.5 + [eye_dist * 0.1, ry * 0.2],
            kps[2] - [0, eye_dist * 0.2]
        ], dtype=np.float32)
        all_glasses_pts = np.vstack([pts0, bridge, pts1])
        hull = cv2.convexHull(all_glasses_pts.astype(np.float32)).reshape(-1, 2)
        glasses_polygon = hull.astype(np.int32)

    if glasses_polygon is not None and len(glasses_polygon) >= 3:
        min_x = int(np.floor(np.min(glasses_polygon[:, 0])))
        max_x = int(np.ceil(np.max(glasses_polygon[:, 0])))
        min_y = int(np.floor(np.min(glasses_polygon[:, 1])))
        max_y = int(np.ceil(np.max(glasses_polygon[:, 1])))

        pad_x = int((max_x - min_x) * 0.08)
        pad_y = int((max_y - min_y) * 0.08)
        min_x = max(0, min_x - pad_x)
        min_y = max(0, min_y - pad_y)
        max_x = min(frame.shape[1], max_x + pad_x)
        max_y = min(frame.shape[0], max_y + pad_y)

        if max_x > min_x and max_y > min_y:
            mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
            poly_roi = glasses_polygon - [min_x, min_y]
            cv2.fillPoly(mask_roi, [poly_roi.astype(np.int32)], 255)
            feather_k = max(3, (min(max_x - min_x, max_y - min_y) // 12) | 1)
            mask_roi = gpu_gaussian_blur(mask_roi, (feather_k, feather_k), feather_k // 3)
            mask[min_y:max_y, min_x:max_x] = mask_roi
            glasses_cutout = frame[min_y:max_y, min_x:max_x].copy()
            glasses_box = (min_x, min_y, max_x, max_y)

    return mask, glasses_cutout, glasses_box, glasses_polygon


def apply_glasses_area(
    frame: np.ndarray,
    glasses_cutout: np.ndarray,
    glasses_box: tuple,
    face_mask: np.ndarray,
    glasses_polygon: np.ndarray,
) -> np.ndarray:
    if frame is None or glasses_cutout is None or glasses_box is None or glasses_polygon is None:
        return frame
    if glasses_cutout.size == 0 or len(glasses_polygon) < 3:
        return frame

    try:
        min_x, min_y, max_x, max_y = map(int, glasses_box)
        frame_h, frame_w = frame.shape[:2]
        min_x = max(0, min(frame_w, min_x))
        min_y = max(0, min(frame_h, min_y))
        max_x = max(0, min(frame_w, max_x))
        max_y = max(0, min(frame_h, max_y))
        box_w = max_x - min_x
        box_h = max_y - min_y
        if box_w <= 0 or box_h <= 0:
            return frame

        roi = frame[min_y:max_y, min_x:max_x]
        if roi.size == 0:
            return frame

        if roi.shape[:2] != glasses_cutout.shape[:2]:
            resized_cutout = gpu_resize(glasses_cutout, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_cutout = glasses_cutout

        poly_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        poly_rel = glasses_polygon - [min_x, min_y]
        cv2.fillPoly(poly_mask, [poly_rel.astype(np.int32)], 255)

        feather_amount = max(1, min(20, min(box_w, box_h) // 10))
        ksize = 2 * feather_amount + 1
        feathered = cv2.GaussianBlur(poly_mask.astype(np.float32), (ksize, ksize), 0)
        max_val = feathered.max()
        if max_val > 1e-6:
            feathered /= max_val
        else:
            feathered.fill(0.0)

        mask_3ch = feathered[:, :, np.newaxis]
        inv_mask = 1.0 - mask_3ch
        blended = resized_cutout.astype(np.float32) * mask_3ch + roi.astype(np.float32) * inv_mask
        frame[min_y:max_y, min_x:max_x] = np.clip(blended, 0, 255).astype(np.uint8)
    except Exception:
        pass

    return frame


def draw_glasses_mask_visualization(
    frame: Frame, face: Face, mask_data: tuple
) -> Frame:
    mask, cutout, (min_x, min_y, max_x, max_y), polygon = mask_data
    if polygon is None or len(polygon) < 3:
        return frame

    vis_frame = frame.copy()
    try:
        cv2.polylines(vis_frame, [polygon.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
        label_y = max(20, min_y - 10)
        cv2.putText(
            vis_frame, "Glasses Mask", (max(0, min_x), label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )
    except Exception:
        pass
    return vis_frame


def create_bangs_mask(
    face: Face, frame: Frame
) -> tuple:
    """Create a mask covering upper forehead / bangs region for hair preservation."""
    if face is None or frame is None:
        return np.zeros((1, 1), dtype=np.uint8), None, (0, 0, 0, 0), None

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    bangs_cutout = None
    bangs_polygon = None
    bangs_box = (0, 0, 0, 0)

    forehead_cutoff = getattr(modules.globals, "forehead_cutoff", 0.0)
    cutoff_ratio = min(1.0, max(0.1, forehead_cutoff / 100.0 if forehead_cutoff > 0 else 0.5))

    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is not None and isinstance(landmarks, np.ndarray) and landmarks.shape[0] >= 106:
        r_brow = landmarks[43:52].astype(np.float32)
        l_brow = landmarks[97:106].astype(np.float32)
        eyebrows = np.vstack([r_brow, l_brow])
        chin = landmarks[16].astype(np.float32)
        brow_center = np.mean(eyebrows, axis=0)

        up_vec = brow_center - chin
        norm = float(np.linalg.norm(up_vec))
        if norm > 1e-3:
            up_vec /= norm
            hairline_offset = up_vec * (norm * 0.75)
            cutoff_offset = up_vec * (norm * (1.0 - cutoff_ratio) * 0.6)
            top_points = eyebrows + hairline_offset
            bottom_points = eyebrows + cutoff_offset
            poly = np.vstack([bottom_points, top_points[::-1]])
            hull = cv2.convexHull(poly.astype(np.float32)).reshape(-1, 2)
            bangs_polygon = hull.astype(np.int32)
    elif hasattr(face, "kps") and face.kps is not None and getattr(face.kps, "shape", None) == (5, 2):
        kps = face.kps.astype(np.float32)
        eye_mid = (kps[0] + kps[1]) * 0.5
        eye_dist = max(10.0, float(np.linalg.norm(kps[1] - kps[0])))
        mouth_mid = (kps[3] + kps[4]) * 0.5
        up_vec = eye_mid - mouth_mid
        norm = float(np.linalg.norm(up_vec))
        if norm > 1e-3:
            up_vec /= norm
            top_mid = eye_mid + up_vec * (eye_dist * 0.8)
            bot_mid = eye_mid + up_vec * (eye_dist * (1.0 - cutoff_ratio) * 0.5)
            w = eye_dist * 1.2
            poly = np.array([
                bot_mid - [w * 0.5, 0],
                bot_mid + [w * 0.5, 0],
                top_mid + [w * 0.6, 0],
                top_mid - [w * 0.6, 0],
            ], dtype=np.float32)
            hull = cv2.convexHull(poly).reshape(-1, 2)
            bangs_polygon = hull.astype(np.int32)

    if bangs_polygon is not None and len(bangs_polygon) >= 3:
        min_x = max(0, int(np.floor(np.min(bangs_polygon[:, 0]))))
        max_x = min(frame.shape[1], int(np.ceil(np.max(bangs_polygon[:, 0]))))
        min_y = max(0, int(np.floor(np.min(bangs_polygon[:, 1]))))
        max_y = min(frame.shape[0], int(np.ceil(np.max(bangs_polygon[:, 1]))))

        if max_x > min_x and max_y > min_y:
            mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
            poly_roi = bangs_polygon - [min_x, min_y]
            cv2.fillPoly(mask_roi, [poly_roi.astype(np.int32)], 255)
            feather_k = max(3, (min(max_x - min_x, max_y - min_y) // 10) | 1)
            mask_roi = gpu_gaussian_blur(mask_roi, (feather_k, feather_k), feather_k // 3)
            mask[min_y:max_y, min_x:max_x] = mask_roi
            bangs_cutout = frame[min_y:max_y, min_x:max_x].copy()
            bangs_box = (min_x, min_y, max_x, max_y)

    return mask, bangs_cutout, bangs_box, bangs_polygon


def apply_bangs_area(
    frame: np.ndarray,
    bangs_cutout: np.ndarray,
    bangs_box: tuple,
    face_mask: np.ndarray,
    bangs_polygon: np.ndarray,
) -> np.ndarray:
    if frame is None or bangs_cutout is None or bangs_box is None or bangs_polygon is None:
        return frame
    if bangs_cutout.size == 0 or len(bangs_polygon) < 3:
        return frame

    try:
        min_x, min_y, max_x, max_y = map(int, bangs_box)
        frame_h, frame_w = frame.shape[:2]
        min_x = max(0, min(frame_w, min_x))
        min_y = max(0, min(frame_h, min_y))
        max_x = max(0, min(frame_w, max_x))
        max_y = max(0, min(frame_h, max_y))
        box_w = max_x - min_x
        box_h = max_y - min_y
        if box_w <= 0 or box_h <= 0:
            return frame

        roi = frame[min_y:max_y, min_x:max_x]
        if roi.size == 0:
            return frame

        if roi.shape[:2] != bangs_cutout.shape[:2]:
            resized_cutout = gpu_resize(bangs_cutout, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_cutout = bangs_cutout

        poly_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        poly_rel = bangs_polygon - [min_x, min_y]
        cv2.fillPoly(poly_mask, [poly_rel.astype(np.int32)], 255)

        feather_amount = max(1, min(25, min(box_w, box_h) // 8))
        ksize = 2 * feather_amount + 1
        feathered = cv2.GaussianBlur(poly_mask.astype(np.float32), (ksize, ksize), 0)
        max_val = feathered.max()
        if max_val > 1e-6:
            feathered /= max_val
        else:
            feathered.fill(0.0)

        mask_3ch = feathered[:, :, np.newaxis]
        inv_mask = 1.0 - mask_3ch
        blended = resized_cutout.astype(np.float32) * mask_3ch + roi.astype(np.float32) * inv_mask
        frame[min_y:max_y, min_x:max_x] = np.clip(blended, 0, 255).astype(np.uint8)
    except Exception:
        pass

    return frame


def draw_bangs_mask_visualization(
    frame: Frame, face: Face, mask_data: tuple
) -> Frame:
    mask, cutout, (min_x, min_y, max_x, max_y), polygon = mask_data
    if polygon is None or len(polygon) < 3:
        return frame

    vis_frame = frame.copy()
    try:
        cv2.polylines(vis_frame, [polygon.astype(np.int32)], True, (0, 255, 128), 2, cv2.LINE_AA)
        label_y = max(20, min_y - 10)
        cv2.putText(
            vis_frame, "Bangs Cutoff", (max(0, min_x), label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA
        )
    except Exception:
        pass
    return vis_frame