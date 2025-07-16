import torch
import numpy as np
import cv2
from PIL import Image
import math
from comfy.utils import ProgressBar

class SBS_VR_Panorama_by_SamSeen:
    """
    Create VR-compatible stereoscopic 360-degree panoramas from equirectangular images.
    Generates depth-aware panoramic content for VR headsets and 360 video players.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from depth_estimator import DepthEstimator
            self.depth_model = DepthEstimator()
        except ImportError:
            self.depth_model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama_image": ("IMAGE", {"tooltip": "Input 360° panoramic image in equirectangular format (2:1 aspect ratio)"}),
                "depth_scale": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 50.0, "step": 0.5, "tooltip": "Controls the intensity of the 3D stereo effect. Higher values = more dramatic depth, but may cause eye strain in VR. Recommended: 10-20 for VR comfort."}),
                "blur_radius": ("INT", {"default": 5, "min": 1, "max": 31, "step": 2, "tooltip": "Smooths depth transitions for comfortable VR viewing. Higher values = smoother but less detailed depth. Odd numbers only. Recommended: 3-7."}),
                "ipd_mm": ("FLOAT", {"default": 65.0, "min": 50.0, "max": 80.0, "step": 1.0, "tooltip": "Interpupillary distance (distance between your eyes) in millimeters. Affects stereo separation. Average adult: 62-68mm. Adjust if VR feels uncomfortable."}),
                "format": (["side_by_side", "over_under"], {"default": "over_under", "tooltip": "VR output format. Over-under works best with Quest 3/Meta headsets. Side-by-side for general VR compatibility."}),
                "invert_depth": ("BOOLEAN", {"default": False, "tooltip": "Reverses depth perception (swap foreground/background). Enable if depth appears backwards in VR."}),
                "depth_quality": (["standard", "high", "ultra"], {"default": "high", "tooltip": "Depth processing quality. Standard=fast, High=balanced, Ultra=best quality with advanced enhancements. Ultra may be slower."}),
                "edge_enhancement": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Enhances fine depth details using unsharp masking. 0=off, 0.2-0.4=recommended, 1.0=maximum enhancement. Higher values may introduce artifacts."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("vr_panorama", "depth_panorama")
    FUNCTION = "create_vr_panorama"
    CATEGORY = "👀 SamSeen"
    DESCRIPTION = "Create VR-compatible stereoscopic 360° panoramas with automatic depth generation for immersive VR experiences"

    def equirectangular_to_perspective(self, equirect_img, fov=90, theta=0, phi=0, width=512, height=512):
        """Convert equirectangular projection to perspective view"""
        eq_h, eq_w = equirect_img.shape[:2]
        
        # Create coordinate arrays
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        x_grid, y_grid = np.meshgrid(x, y)
        
        # Convert to spherical coordinates
        r = np.sqrt(x_grid**2 + y_grid**2)
        valid_mask = r <= 1
        
        # Calculate angles
        lat = np.arcsin(y_grid * np.sin(fov/2 * np.pi/180)) + phi
        lon = np.arctan2(x_grid, np.cos(fov/2 * np.pi/180)) + theta
        
        # Convert to equirectangular coordinates
        eq_x = ((lon + np.pi) / (2 * np.pi) * eq_w).astype(np.int32)
        eq_y = ((lat + np.pi/2) / np.pi * eq_h).astype(np.int32)
        
        # Clamp coordinates
        eq_x = np.clip(eq_x, 0, eq_w - 1)
        eq_y = np.clip(eq_y, 0, eq_h - 1)
        
        # Sample from equirectangular image
        if len(equirect_img.shape) == 3:
            result = equirect_img[eq_y, eq_x]
        else:
            result = equirect_img[eq_y, eq_x]
        
        # Apply mask for circular FOV
        if len(result.shape) == 3:
            result[~valid_mask] = [0, 0, 0]
        else:
            result[~valid_mask] = 0
            
        return result

    def perspective_to_equirectangular(self, perspective_img, theta, phi, eq_width, eq_height, fov=90):
        """Convert perspective view back to equirectangular coordinates"""
        return perspective_img

    def enhance_depth_quality(self, depth_map, quality_level="high", edge_enhancement=0.0):
        """Enhance depth map quality using various techniques"""
        
        if quality_level == "standard":
            return depth_map
            
        enhanced_depth = depth_map.copy()
        
        # High quality enhancements
        if quality_level in ["high", "ultra"]:
            
            # Edge enhancement using unsharp masking
            if edge_enhancement > 0.0:
                # Create a blurred version
                blurred = cv2.GaussianBlur(enhanced_depth, (5, 5), 0)
                # Subtract to get high-frequency details
                mask = enhanced_depth - blurred
                # Add back scaled high-frequency details
                enhanced_depth = enhanced_depth + (mask * edge_enhancement)
                # Clamp to valid range
                enhanced_depth = np.clip(enhanced_depth, 0.0, 1.0)
                print(f"Applied edge enhancement: {edge_enhancement}")

            # Increase contrast while preserving detail
            mean_depth = np.mean(enhanced_depth)
            enhanced_depth = np.clip((enhanced_depth - mean_depth) * 1.2 + mean_depth, 0.0, 1.0)
            
            # Bilateral filter to smooth while preserving edges
            enhanced_depth_8bit = (enhanced_depth * 255).astype(np.uint8)
            filtered = cv2.bilateralFilter(enhanced_depth_8bit, 9, 75, 75)
            enhanced_depth = filtered.astype(np.float32) / 255.0
            
        # Ultra quality additional enhancements
        if quality_level == "ultra":
            
            # Process at multiple scales and combine
            h, w = enhanced_depth.shape
            
            # Scale down, process, scale up
            small_depth = cv2.resize(enhanced_depth, (w//2, h//2), interpolation=cv2.INTER_AREA)
            small_enhanced = cv2.bilateralFilter((small_depth * 255).astype(np.uint8), 5, 50, 50).astype(np.float32) / 255.0
            upscaled = cv2.resize(small_enhanced, (w, h), interpolation=cv2.INTER_CUBIC)
            
            # Blend with original using edge-aware weights
            edges = cv2.Canny((enhanced_depth * 255).astype(np.uint8), 50, 150)
            edge_mask = (edges > 0).astype(np.float32)
            edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0)
            
            # More original detail in edge areas, more smoothed in flat areas
            enhanced_depth = enhanced_depth * edge_mask + upscaled * (1 - edge_mask)
            
            print("Applied ultra quality multi-scale refinement")
        
        print(f"Depth enhancement complete: quality={quality_level}, edge_enhancement={edge_enhancement}")
        print(f"Enhanced depth: min={np.min(enhanced_depth)}, max={np.max(enhanced_depth)}, mean={np.mean(enhanced_depth)}")
        
        return enhanced_depth

    def generate_panorama_depth(self, panorama_tensor, blur_radius, quality_level="high"):
        """Generate depth map using the EXACT same logic as SBS V2 node with quality enhancements"""
        if self.depth_model is None:
            try:
                from depth_estimator import DepthEstimator
                self.depth_model = DepthEstimator()
                self.depth_model.load_model()
            except Exception as e:
                print(f"Error loading depth model: {e}")
                h, w = panorama_tensor.shape[1:3]
                return np.ones((h, w), dtype=np.float32) * 0.5

        try:
            # Set the blur radius
            self.depth_model.blur_radius = blur_radius
            
            # Process as tensor like SBS V2 - convert to numpy 
            panorama_np = panorama_tensor.cpu().numpy() * 255.0
            panorama_np = panorama_np.astype(np.uint8)
            
            print(f"Processing panorama with Depth Anything V2: {panorama_np.shape}")
            
            # For higher quality, we can try processing at higher resolution
            original_h, original_w = panorama_np.shape[:2]
            
            if quality_level == "ultra" and original_w < 8192:
                # Upscale for processing if not already at high resolution
                scale_factor = min(2.0, 8192 / original_w)
                if scale_factor > 1.1:  # Only upscale if meaningful
                    new_w = int(original_w * scale_factor)
                    new_h = int(original_h * scale_factor)
                    upscaled = cv2.resize(panorama_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    print(f"Upscaled for processing: {panorama_np.shape} -> {upscaled.shape}")
                    
                    # Process at higher resolution
                    depth_map = self.depth_model.predict_depth(upscaled)
                    
                    # Downscale back to original resolution
                    depth_map = cv2.resize(depth_map, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
                    print(f"Downscaled back to: {depth_map.shape}")
                else:
                    # Process at original resolution
                    depth_map = self.depth_model.predict_depth(panorama_np)
            else:
                # Standard processing
                depth_map = self.depth_model.predict_depth(panorama_np)
            
            print(f"Raw depth output: shape={depth_map.shape}, min={np.min(depth_map)}, max={np.max(depth_map)}, mean={np.mean(depth_map)}")
            
            # Apply the same normalization as SBS V2
            if np.min(depth_map) < 0 or np.max(depth_map) > 1:
                depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
            
            print(f"Normalized depth: min={np.min(depth_map)}, max={np.max(depth_map)}, mean={np.mean(depth_map)}")
            
            return depth_map
            
        except Exception as e:
            print(f"Error generating depth map: {e}")
            h, w = panorama_tensor.shape[1:3]
            return np.ones((h, w), dtype=np.float32) * 0.5

    def create_stereo_displacement(self, depth_map, ipd_mm, format_type):
        """Create stereo displacement based on depth and interpupillary distance for panoramic content"""
        h, w = depth_map.shape
        
        # Convert IPD from mm to angular displacement for panoramic content
        # This accounts for the fact that we're dealing with spherical projection
        ipd_angular = (ipd_mm / 1000.0) * (w / (2 * np.pi))  # Convert to pixels in panoramic space
        
        # Create displacement map that varies by depth
        # Closer objects (higher depth values) get more displacement
        # Scale down for panoramic content to avoid excessive shifts
        displacement = depth_map * ipd_angular * 0.3  # Reduced scaling for panoramic
        
        return displacement

    def apply_stereo_shift(self, image, displacement, eye='left'):
        """Apply horizontal shift with proper panoramic coordinate handling"""
        h, w = image.shape[:2]
        
        # Stereo direction (back to original)
        if eye == 'left':
            shift_factor = -1.0
        else:  # right eye
            shift_factor = 1.0
        
        shifted_image = np.zeros_like(image)
        
        # Process each row
        for y in range(h):
            # Calculate latitude factor (less displacement at poles)
            lat_factor = np.cos((y / h - 0.5) * np.pi)  # Ranges from 0 at poles to 1 at equator
            
            for x in range(w):
                # Calculate shift amount adjusted for latitude
                shift = displacement[y, x] * shift_factor * lat_factor
                source_x = x - int(shift)
                
                # Handle wraparound properly
                source_x = source_x % w
                
                # Bilinear interpolation for smoother results
                x_floor = int(source_x) % w
                x_ceil = (x_floor + 1) % w
                x_frac = source_x - int(source_x)
                
                if len(image.shape) == 3:  # Color image
                    pixel_value = (1 - x_frac) * image[y, x_floor] + x_frac * image[y, x_ceil]
                else:  # Grayscale
                    pixel_value = (1 - x_frac) * image[y, x_floor] + x_frac * image[y, x_ceil]
                
                shifted_image[y, x] = pixel_value
        
        return shifted_image

    def create_vr_panorama(self, panorama_image, depth_scale, blur_radius, ipd_mm, format, invert_depth, depth_quality="high", edge_enhancement=0.0):
        """Create VR-compatible stereoscopic panorama"""
        
        # Handle batch dimension properly - but keep as tensor for depth processing
        if len(panorama_image.shape) == 4:
            # Batch of images - process first image but keep as tensor
            panorama_tensor = panorama_image[0:1]  # Keep batch dimension for consistency
        else:
            # Single image - add batch dimension
            panorama_tensor = panorama_image.unsqueeze(0)
        
        print(f"Processing panorama tensor: {panorama_tensor.shape}")
        
        # Generate depth map using SBS V2 approach with quality enhancements
        print(f"Generating panoramic depth map using exact SBS V2 approach with {depth_quality} quality...")
        depth_map = self.generate_panorama_depth(panorama_tensor[0], blur_radius, depth_quality)
        
        # Apply quality enhancements
        if depth_quality != "standard" or edge_enhancement > 0.0:
            depth_map = self.enhance_depth_quality(depth_map, depth_quality, edge_enhancement)
        
        # Convert tensor to numpy for stereo processing
        panorama_np = panorama_tensor[0].cpu().numpy() * 255.0
        panorama_np = panorama_np.astype(np.uint8)
        
        # Apply blur to depth map (same as SBS V2)
        if blur_radius > 1:
            # Make sure blur_radius is odd
            if blur_radius % 2 == 0:
                blur_radius += 1
            depth_map = cv2.GaussianBlur(depth_map, (blur_radius, blur_radius), 0)
            print(f"Applied Gaussian blur with radius {blur_radius}")
        
        if invert_depth:
            print("Applying user-requested depth inversion")
            depth_map = 1.0 - depth_map
        
        # Scale depth
        depth_map = depth_map * (depth_scale / 100.0)
        
        # Create displacement map
        displacement = self.create_stereo_displacement(depth_map, ipd_mm, format)
        
        # Generate left and right eye views
        print("Generating stereo views...")
        left_eye = self.apply_stereo_shift(panorama_np, displacement, 'left')
        right_eye = self.apply_stereo_shift(panorama_np, displacement, 'right')
        
        # Combine into stereo format
        h, w = panorama_np.shape[:2]
        
        if format == "side_by_side":
            # Side-by-side format (left|right)
            stereo_panorama = np.zeros((h, w * 2, 3), dtype=np.uint8)
            stereo_panorama[:, :w] = left_eye
            stereo_panorama[:, w:] = right_eye
        else:  # over_under
            # Over-under format (top=left, bottom=right)
            stereo_panorama = np.zeros((h * 2, w, 3), dtype=np.uint8)
            stereo_panorama[:h, :] = left_eye
            stereo_panorama[h:, :] = right_eye
        
        # Convert back to tensors
        stereo_tensor = torch.tensor(stereo_panorama.astype(np.float32) / 255.0).unsqueeze(0)
        
        # Create depth visualization
        depth_vis = np.stack([depth_map, depth_map, depth_map], axis=-1)
        depth_tensor = torch.tensor(depth_vis).unsqueeze(0)
        
        print(f"Created VR panorama: {stereo_tensor.shape}")
        
        return (stereo_tensor, depth_tensor)
