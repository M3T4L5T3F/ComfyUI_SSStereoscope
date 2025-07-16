import torch
import numpy as np
import cv2
from PIL import Image
import math
from comfy.utils import ProgressBar
import gc
from typing import Tuple, Optional

class SBS_VR_Panorama_by_SamSeen:
    """
    Create VR-compatible stereoscopic 360-degree panoramas from equirectangular images.
    Generates depth-aware panoramic content for VR headsets and 360 video players.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = None
        self._depth_cache = {}
        
        # Pre-computed constants for optimization
        self._pi = math.pi
        self._two_pi = 2 * math.pi
        
    def _load_depth_model(self):
        """Lazy load depth model to save memory"""
        if self.depth_model is None:
            try:
                from depth_estimator import DepthEstimator
                self.depth_model = DepthEstimator()
                self.depth_model.load_model()
                print("Loaded depth model for VR panorama processing")
            except Exception as e:
                print(f"Warning: Could not load depth model: {e}")
                return False
        return True

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
                "depth_consistency": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Temporal consistency for depth maps. Higher values create smoother depth transitions across the panorama."}),
                "memory_optimization": ("BOOLEAN", {"default": True, "tooltip": "Enable memory optimization for large panoramas. Recommended for 4K+ images."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("vr_panorama", "depth_panorama")
    FUNCTION = "create_vr_panorama"
    CATEGORY = "👀 SamSeen"
    DESCRIPTION = "Create VR-compatible stereoscopic 360° panoramas with automatic depth generation for immersive VR experiences"

    def _compute_optimal_tile_size(self, width: int, height: int, max_memory_mb: int = 2048) -> Tuple[int, int]:
        """Compute optimal tile size for memory-efficient processing"""
        # Estimate memory usage per pixel (RGB + depth + intermediate calculations)
        bytes_per_pixel = 4 * 3 + 4 + 4 * 2  # ~20 bytes per pixel
        max_pixels = (max_memory_mb * 1024 * 1024) // bytes_per_pixel
        
        if width * height <= max_pixels:
            return width, height
            
        # Calculate tile dimensions maintaining aspect ratio
        aspect_ratio = width / height
        tile_height = int(math.sqrt(max_pixels / aspect_ratio))
        tile_width = int(tile_height * aspect_ratio)
        
        # Ensure tiles are divisible by 14 (patch size for ViT)
        tile_width = (tile_width // 14) * 14
        tile_height = (tile_height // 14) * 14
        
        return max(tile_width, 14), max(tile_height, 14)

    def _process_panorama_tiles(self, panorama_np: np.ndarray, tile_width: int, tile_height: int, 
                               overlap: int = 56) -> np.ndarray:
        """Process panorama in tiles with overlap for seamless depth generation"""
        height, width = panorama_np.shape[:2]
        depth_map = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        # Calculate number of tiles
        num_tiles_x = math.ceil(width / (tile_width - overlap))
        num_tiles_y = math.ceil(height / (tile_height - overlap))
        
        print(f"Processing {num_tiles_x}x{num_tiles_y} tiles of size {tile_width}x{tile_height}")
        
        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                # Calculate tile boundaries
                start_x = tx * (tile_width - overlap)
                start_y = ty * (tile_height - overlap)
                end_x = min(start_x + tile_width, width)
                end_y = min(start_y + tile_height, height)
                
                # Extract tile with padding if at edges
                tile = panorama_np[start_y:end_y, start_x:end_x]
                
                # Pad tile to required size if needed
                if tile.shape[0] < tile_height or tile.shape[1] < tile_width:
                    padded_tile = np.zeros((tile_height, tile_width, 3), dtype=panorama_np.dtype)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile
                
                # Process tile
                try:
                    tile_depth = self.depth_model.predict_depth(tile)
                    tile_depth = tile_depth[:end_y-start_y, :end_x-start_x]  # Remove padding
                except Exception as e:
                    print(f"Error processing tile ({tx}, {ty}): {e}")
                    tile_depth = np.ones((end_y-start_y, end_x-start_x), dtype=np.float32) * 0.5
                
                # Create weight map for blending (higher weights in center)
                tile_weights = np.ones((end_y-start_y, end_x-start_x), dtype=np.float32)
                if overlap > 0:
                    # Apply cosine weighting for smooth blending
                    for i in range(tile_weights.shape[0]):
                        for j in range(tile_weights.shape[1]):
                            weight_x = 1.0
                            weight_y = 1.0
                            
                            if start_x > 0 and j < overlap // 2:
                                weight_x = 0.5 * (1 + math.cos(math.pi * (overlap // 2 - j) / (overlap // 2)))
                            elif end_x < width and j >= tile_weights.shape[1] - overlap // 2:
                                weight_x = 0.5 * (1 + math.cos(math.pi * (j - (tile_weights.shape[1] - overlap // 2)) / (overlap // 2)))
                                
                            if start_y > 0 and i < overlap // 2:
                                weight_y = 0.5 * (1 + math.cos(math.pi * (overlap // 2 - i) / (overlap // 2)))
                            elif end_y < height and i >= tile_weights.shape[0] - overlap // 2:
                                weight_y = 0.5 * (1 + math.cos(math.pi * (i - (tile_weights.shape[0] - overlap // 2)) / (overlap // 2)))
                            
                            tile_weights[i, j] = weight_x * weight_y
                
                # Accumulate depth and weights
                depth_map[start_y:end_y, start_x:end_x] += tile_depth * tile_weights
                weight_map[start_y:end_y, start_x:end_x] += tile_weights
                
                # Clean up to save memory
                del tile_depth, tile_weights
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Normalize by weights
        valid_mask = weight_map > 0
        depth_map[valid_mask] /= weight_map[valid_mask]
        
        # Fill any remaining zero areas
        if not np.all(valid_mask):
            print("Warning: Some areas not covered by tiles, filling with interpolation")
            from scipy.interpolate import griddata
            valid_points = np.column_stack(np.where(valid_mask))
            valid_values = depth_map[valid_mask]
            invalid_points = np.column_stack(np.where(~valid_mask))
            
            if len(valid_points) > 3 and len(invalid_points) > 0:
                interpolated = griddata(valid_points, valid_values, invalid_points, method='linear', fill_value=0.5)
                depth_map[~valid_mask] = interpolated
        
        return depth_map

    def _enhance_panoramic_depth(self, depth_map: np.ndarray, quality_level: str = "high", 
                                edge_enhancement: float = 0.0, consistency: float = 0.3) -> np.ndarray:
        """Enhanced depth processing specifically optimized for panoramic content"""
        
        if quality_level == "standard":
            return depth_map
            
        enhanced_depth = depth_map.copy()
        h, w = enhanced_depth.shape
        
        # 1. Panoramic pole correction - reduce distortion at top/bottom
        if quality_level in ["high", "ultra"]:
            pole_correction_strength = 0.3
            for y in range(h):
                # Calculate distance from equator (0.5)
                lat_norm = abs(y / h - 0.5) * 2  # 0 at equator, 1 at poles
                if lat_norm > 0.7:  # Only correct near poles
                    correction_factor = 1.0 - (lat_norm - 0.7) * pole_correction_strength
                    enhanced_depth[y, :] *= correction_factor
        
        # 2. Circular continuity enforcement for 360° wrapping
        if consistency > 0.0:
            # Ensure left and right edges match (360° continuity)
            edge_width = min(w // 20, 50)  # Use up to 50 pixels or 5% of width
            left_edge = enhanced_depth[:, :edge_width]
            right_edge = enhanced_depth[:, -edge_width:]
            
            # Blend edges for seamless wrapping
            for i in range(edge_width):
                weight = (i + 1) / edge_width * consistency
                blended_left = (1 - weight) * left_edge[:, i] + weight * right_edge[:, -(i+1)]
                blended_right = (1 - weight) * right_edge[:, -(i+1)] + weight * left_edge[:, i]
                
                enhanced_depth[:, i] = blended_left
                enhanced_depth[:, -(i+1)] = blended_right
        
        # 3. Edge enhancement with panoramic awareness
        if edge_enhancement > 0.0:
            # Use adaptive kernel size based on image resolution
            kernel_size = max(3, min(15, w // 512))
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Create latitude-weighted enhancement
            blurred = cv2.GaussianBlur(enhanced_depth, (kernel_size, kernel_size), 0)
            edge_mask = enhanced_depth - blurred
            
            # Apply latitude weighting (stronger enhancement at equator)
            lat_weights = np.zeros(h)
            for y in range(h):
                lat_norm = abs(y / h - 0.5) * 2
                lat_weights[y] = 1.0 - lat_norm * 0.5  # Reduce enhancement at poles
            
            for y in range(h):
                enhanced_depth[y, :] += edge_mask[y, :] * edge_enhancement * lat_weights[y]
            
            enhanced_depth = np.clip(enhanced_depth, 0.0, 1.0)
        
        # 4. Ultra quality: Multi-scale refinement
        if quality_level == "ultra":
            # Process at multiple scales and combine
            scales = [1.0, 0.5, 0.25]
            scale_weights = [0.6, 0.3, 0.1]
            refined_depth = np.zeros_like(enhanced_depth)
            
            for scale, weight in zip(scales, scale_weights):
                if scale == 1.0:
                    scale_depth = enhanced_depth
                else:
                    scale_w, scale_h = int(w * scale), int(h * scale)
                    scale_depth = cv2.resize(enhanced_depth, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
                    
                    # Apply bilateral filter at this scale
                    scale_depth_8bit = (scale_depth * 255).astype(np.uint8)
                    filtered = cv2.bilateralFilter(scale_depth_8bit, 9, 75, 75)
                    scale_depth = filtered.astype(np.float32) / 255.0
                    
                    # Resize back to original size
                    scale_depth = cv2.resize(scale_depth, (w, h), interpolation=cv2.INTER_CUBIC)
                
                refined_depth += scale_depth * weight
            
            enhanced_depth = refined_depth
        
        # 5. Final normalization and contrast enhancement
        mean_depth = np.mean(enhanced_depth)
        enhanced_depth = np.clip((enhanced_depth - mean_depth) * 1.1 + mean_depth, 0.0, 1.0)
        
        print(f"Depth enhancement complete: quality={quality_level}, edge={edge_enhancement}, consistency={consistency}")
        
        return enhanced_depth

    def _generate_panorama_depth(self, panorama_tensor: torch.Tensor, blur_radius: int, 
                               quality_level: str = "high", memory_optimization: bool = True) -> np.ndarray:
        """Generate depth map with memory optimization for large panoramas"""
        
        if not self._load_depth_model():
            h, w = panorama_tensor.shape[1:3]
            print("Using fallback depth map")
            return self._create_fallback_depth(h, w)
        
        try:
            # Set blur radius
            self.depth_model.blur_radius = blur_radius
            
            # Convert to numpy
            panorama_np = panorama_tensor.cpu().numpy() * 255.0
            panorama_np = panorama_np.astype(np.uint8)
            
            original_h, original_w = panorama_np.shape[:2]
            print(f"Processing panorama: {original_w}x{original_h}")
            
            # Memory optimization: process in tiles for large images
            if memory_optimization and (original_w > 4096 or original_h > 2048):
                tile_w, tile_h = self._compute_optimal_tile_size(original_w, original_h)
                print(f"Using tiled processing: {tile_w}x{tile_h} tiles")
                depth_map = self._process_panorama_tiles(panorama_np, tile_w, tile_h)
            else:
                # Process entire image
                depth_map = self.depth_model.predict_depth(panorama_np)
            
            # Normalize depth map
            if np.min(depth_map) < 0 or np.max(depth_map) > 1:
                depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
            
            return depth_map
            
        except Exception as e:
            print(f"Error generating depth map: {e}")
            return self._create_fallback_depth(panorama_tensor.shape[1], panorama_tensor.shape[2])

    def _create_fallback_depth(self, height: int, width: int) -> np.ndarray:
        """Create a reasonable fallback depth map for panoramas"""
        # Create depth based on distance from center with some variation
        depth = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                # Convert to spherical coordinates
                lat = (y / height - 0.5) * math.pi
                lon = (x / width - 0.5) * 2 * math.pi
                
                # Create depth variation based on spherical coordinates
                depth_val = 0.5 + 0.3 * math.cos(lat * 2) + 0.2 * math.sin(lon * 3)
                depth[y, x] = np.clip(depth_val, 0.0, 1.0)
        
        return depth

    def _create_stereo_displacement_optimized(self, depth_map: np.ndarray, ipd_mm: float) -> np.ndarray:
        """Optimized stereo displacement calculation for panoramic content"""
        h, w = depth_map.shape
        
        # Convert IPD to angular displacement (optimized for VR viewing)
        ipd_angular = (ipd_mm / 1000.0) * (w / self._two_pi) * 0.25  # Reduced for comfort
        
        # Pre-compute latitude factors for all rows
        lat_factors = np.cos((np.arange(h) / h - 0.5) * self._pi)
        
        # Vectorized displacement calculation
        displacement = depth_map * ipd_angular
        displacement = displacement * lat_factors[:, np.newaxis]
        
        return displacement

    def _apply_stereo_shift_optimized(self, image: np.ndarray, displacement: np.ndarray, 
                                   eye: str = 'left') -> np.ndarray:
        """Memory-optimized stereo shifting with improved interpolation"""
        h, w = image.shape[:2]
        shift_factor = -1.0 if eye == 'left' else 1.0
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Calculate shifted coordinates
        shifted_x = x_coords - (displacement * shift_factor).astype(np.float32)
        
        # Handle wraparound for panoramic coordinates
        shifted_x = shifted_x % w
        
        # Use OpenCV's remap for efficient interpolation
        map_x = shifted_x.astype(np.float32)
        map_y = y_coords.astype(np.float32)
        
        shifted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        
        return shifted_image

    def create_vr_panorama(self, panorama_image, depth_scale, blur_radius, ipd_mm, format, 
                          invert_depth, depth_quality="high", edge_enhancement=0.0, 
                          depth_consistency=0.3, memory_optimization=True):
        """Create VR-compatible stereoscopic panorama with optimizations"""
        
        # Input validation
        if blur_radius % 2 == 0:
            blur_radius += 1
            
        # Handle batch dimension
        if len(panorama_image.shape) == 4:
            panorama_tensor = panorama_image[0:1]
        else:
            panorama_tensor = panorama_image.unsqueeze(0)
        
        # Check aspect ratio
        h, w = panorama_tensor.shape[1:3]
        aspect_ratio = w / h
        if not (1.8 <= aspect_ratio <= 2.2):
            print(f"Warning: Aspect ratio {aspect_ratio:.2f} is not typical for equirectangular (should be ~2:1)")
        
        print(f"Processing {w}x{h} panorama with {depth_quality} quality, memory_opt={memory_optimization}")
        
        try:
            # Generate depth map
            depth_map = self._generate_panorama_depth(
                panorama_tensor[0], blur_radius, depth_quality, memory_optimization
            )
            
            # Apply enhancements
            depth_map = self._enhance_panoramic_depth(
                depth_map, depth_quality, edge_enhancement, depth_consistency
            )
            
            # Apply blur and depth inversion
            if blur_radius > 1:
                depth_map = cv2.GaussianBlur(depth_map, (blur_radius, blur_radius), 0)
            
            if invert_depth:
                depth_map = 1.0 - depth_map
            
            # Scale depth
            depth_map = depth_map * (depth_scale / 100.0)
            
            # Convert to numpy for processing
            panorama_np = panorama_tensor[0].cpu().numpy() * 255.0
            panorama_np = panorama_np.astype(np.uint8)
            
            # Create optimized displacement map
            displacement = self._create_stereo_displacement_optimized(depth_map, ipd_mm)
            
            # Generate stereo views
            print("Generating stereo views with optimized processing...")
            left_eye = self._apply_stereo_shift_optimized(panorama_np, displacement, 'left')
            right_eye = self._apply_stereo_shift_optimized(panorama_np, displacement, 'right')
            
            # Combine into stereo format
            if format == "side_by_side":
                stereo_panorama = np.concatenate([left_eye, right_eye], axis=1)
            else:  # over_under
                stereo_panorama = np.concatenate([left_eye, right_eye], axis=0)
            
            # Convert back to tensors
            stereo_tensor = torch.tensor(stereo_panorama.astype(np.float32) / 255.0).unsqueeze(0)
            
            # Create depth visualization
            depth_vis = np.stack([depth_map, depth_map, depth_map], axis=-1)
            depth_tensor = torch.tensor(depth_vis).unsqueeze(0)
            
            # Clean up memory
            if memory_optimization:
                del panorama_np, left_eye, right_eye, displacement, depth_map
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"VR panorama created: {stereo_tensor.shape}")
            return (stereo_tensor, depth_tensor)
            
        except Exception as e:
            print(f"Error creating VR panorama: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback result
            h, w = panorama_tensor.shape[1:3]
            if format == "side_by_side":
                fallback_shape = (h, w * 2, 3)
            else:
                fallback_shape = (h * 2, w, 3)
            
            fallback_stereo = torch.zeros(fallback_shape).unsqueeze(0)
            fallback_depth = torch.zeros((h, w, 3)).unsqueeze(0)
            
            return (fallback_stereo, fallback_depth)
