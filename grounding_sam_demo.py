#!/usr/bin/env python3
"""
Independent Grounding SAM Demo Script

This script loads Grounding DINO and SAM models to demonstrate object detection 
and segmentation on the d3fields adamanip dataset. It provides visualization
of both individual frames and batch processing capabilities.

Usage:
    python grounding_sam_demo.py [options]

Examples:
    # Process single image from specific environment
    python grounding_sam_demo.py --scene OpenBottle --env_type grasp_env --env_id 0 --camera_id 0 --frame_id 2

    # Process multiple frames with custom queries
    python grounding_sam_demo.py --scene OpenDoor --queries "door handle" "robotic arm" --batch_process --max_frames 5

    # Save results to custom directory
    python grounding_sam_demo.py --scene OpenCoffeeMachine --output_dir ./demo_results --save_masks
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

# Grounding DINO and SAM imports
import groundingdino
from groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import build_sam, SamPredictor

# Local utilities (assuming they're available)
sys.path.append(os.getcwd())
from utils.grounded_sam import grounded_instance_sam_new_ver


class GroundingSAMDemo:
    """
    Grounding SAM demonstration class for d3fields adamanip dataset
    """
    
    def __init__(self, device: str = "cuda:0"):
        """
        Initialize the Grounding SAM demo
        
        Args:
            device: CUDA device to use
        """
        self.device = device
        self.setup_models()
        
        # Default dataset path
        self.dataset_root = "/ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields"
        
        # Default queries for different scenes
        self.default_queries = {
            'OpenBottle': ['bottle', 'robotic arm'],
            'OpenDoor': ['door', 'robotic arm'],
            'OpenCoffeeMachine': ['coffee machine','robotic arm'],
            'OpenSafe': ['safe', 'robotic arm'],
            'OpenWindow': ['window',  'robotic arm'],
            'OpenPressureCooker': ['pressure cooker', 'robotic arm']
        }
        
        # Default thresholds
        self.default_thresholds = [0.25, 0.3, 0.2, 0.2]

    def setup_models(self):
        """Setup Grounding DINO and SAM models"""
        print("Setting up Grounding DINO and SAM models...")
        
        # Setup paths
        curr_path = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(groundingdino.__path__[0], 'config/GroundingDINO_SwinT_OGC.py')
        grounded_checkpoint = os.path.join(curr_path, 'ckpts/groundingdino_swint_ogc.pth')
        sam_checkpoint = os.path.join(curr_path, 'ckpts/sam_vit_h_4b8939.pth')
        
        # Download models if needed
        self._download_models(grounded_checkpoint, sam_checkpoint)
        
        # Initialize models
        self.ground_dino_model = GroundingDINOModel(config_file, grounded_checkpoint, device=self.device)
        self.sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint))
        self.sam_model.model = self.sam_model.model.to(self.device)
        
        print("Models loaded successfully!")

    def _download_models(self, grounded_checkpoint: str, sam_checkpoint: str):
        """Download model checkpoints if they don't exist"""
        ckpts_dir = os.path.dirname(grounded_checkpoint)
        os.makedirs(ckpts_dir, exist_ok=True)
        
        if not os.path.exists(grounded_checkpoint):
            print('Downloading GroundingDINO model...')
            os.system('wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth')
            os.system(f'mv groundingdino_swint_ogc.pth {ckpts_dir}')
            
        if not os.path.exists(sam_checkpoint):
            print('Downloading SAM model...')
            os.system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')
            os.system(f'mv sam_vit_h_4b8939.pth {ckpts_dir}')

    def get_available_scenes(self) -> List[str]:
        """Get list of available scenes in the dataset"""
        if not os.path.exists(self.dataset_root):
            print(f"Dataset root not found: {self.dataset_root}")
            return []
        
        scenes = [d for d in os.listdir(self.dataset_root) 
                 if os.path.isdir(os.path.join(self.dataset_root, d))]
        return sorted(scenes)

    def get_available_environments(self, scene: str) -> Dict[str, List[int]]:
        """Get available environments for a scene"""
        scene_path = os.path.join(self.dataset_root, scene)
        if not os.path.exists(scene_path):
            return {}
        
        environments = {}
        for env_dir in os.listdir(scene_path):
            env_path = os.path.join(scene_path, env_dir)
            if os.path.isdir(env_path):
                env_type = env_dir.split('_')[0] + '_' + env_dir.split('_')[1]  # e.g., "grasp_env"
                env_id = int(env_dir.split('_')[2])  # e.g., 0
                
                if env_type not in environments:
                    environments[env_type] = []
                environments[env_type].append(env_id)
        
        # Sort environment IDs
        for env_type in environments:
            environments[env_type] = sorted(environments[env_type])
            
        return environments

    def load_image(self, scene: str, env_type: str, env_id: int, camera_id: int, frame_id: int) -> Optional[np.ndarray]:
        """
        Load an image from the dataset
        
        Args:
            scene: Scene name (e.g., 'OpenBottle')
            env_type: Environment type (e.g., 'grasp_env', 'manip_env')
            env_id: Environment ID
            camera_id: Camera ID (0 or 1)
            frame_id: Frame ID
            
        Returns:
            BGR image as numpy array or None if not found
        """
        image_path = os.path.join(
            self.dataset_root, scene, f"{env_type}_{env_id}", 
            f"camera_{camera_id}", "color", f"{frame_id}.png"
        )
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
            
        image = cv2.imread(image_path)
        return image

    def get_available_frames(self, scene: str, env_type: str, env_id: int, camera_id: int) -> List[int]:
        """Get available frame IDs for a specific camera"""
        color_dir = os.path.join(
            self.dataset_root, scene, f"{env_type}_{env_id}", 
            f"camera_{camera_id}", "color"
        )
        
        if not os.path.exists(color_dir):
            return []
            
        frames = []
        for file in os.listdir(color_dir):
            if file.endswith('.png'):
                try:
                    frame_id = int(file.split('.')[0])
                    frames.append(frame_id)
                except ValueError:
                    continue
                    
        return sorted(frames)

    def process_single_image(self, 
                           image: np.ndarray, 
                           queries: List[str], 
                           thresholds: List[float],
                           cam_idx: int = 0,
                           save_individual_masks: bool = False) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Process a single image with Grounding SAM
        
        Args:
            image: BGR image
            queries: List of text queries
            thresholds: List of detection thresholds
            cam_idx: Camera index for saving intermediate results
            save_individual_masks: Whether to save individual detection masks
            
        Returns:
            Tuple of (masks, labels, confidences)
        """
        # Ensure thresholds match queries
        if len(thresholds) == 1:
            thresholds = thresholds * len(queries)
        elif len(thresholds) != len(queries):
            thresholds = [0.25] * len(queries)
            
        # Run Grounding SAM
        masks, labels, confidences = grounded_instance_sam_new_ver(
            image, queries, self.ground_dino_model, self.sam_model, 
            thresholds, merge_all=False, device=self.device, cam_idx=cam_idx
        )
        
        return masks, labels, confidences

    def visualize_results(self, 
                         image: np.ndarray, 
                         masks: np.ndarray, 
                         labels: List[str], 
                         confidences: np.ndarray,
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> np.ndarray:
        """
        Visualize detection and segmentation results
        
        Args:
            image: Original BGR image
            masks: Instance masks
            labels: Object labels
            confidences: Detection confidences
            save_path: Path to save visualization
            show_plot: Whether to display the plot
            
        Returns:
            Visualization image
        """
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Grounding SAM Results', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # All masks overlay
        axes[0, 1].imshow(image_rgb)
        colors = plt.cm.tab10(np.linspace(0, 1, len(masks)))
        
        for i, (mask, label, conf) in enumerate(zip(masks, labels, confidences)):
            if i == 0:  # Skip background
                continue
            color = colors[i % len(colors)]
            
            # Create colored mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = [*color[:3], 0.6]
            axes[0, 1].imshow(colored_mask)
            
            # Add label
            y, x = np.where(mask)
            if len(y) > 0:
                center_y, center_x = np.mean(y), np.mean(x)
                axes[0, 1].text(center_x, center_y, f'{label}\n{conf:.2f}', 
                               color='white', fontsize=10, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        axes[0, 1].set_title('All Detections Overlay')
        axes[0, 1].axis('off')
        
        # Individual masks grid
        if len(masks) > 1:
            # Show first few individual masks
            for i in range(1, min(3, len(masks))):  # Skip background, show up to 2 objects
                row = 1
                col = i - 1
                if col < 2:
                    axes[row, col].imshow(image_rgb)
                    
                    # Overlay single mask
                    mask = masks[i]
                    colored_mask = np.zeros((*mask.shape, 4))
                    colored_mask[mask] = [*colors[i % len(colors)][:3], 0.8]
                    axes[row, col].imshow(colored_mask)
                    
                    axes[row, col].set_title(f'{labels[i]} ({confidences[i]:.2f})')
                    axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(len(masks)-1, 2):
            if i < 2:
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        # Return the figure as an image array
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return vis_image

    def visualize_dual_camera_results(self, 
                                     images: List[np.ndarray], 
                                     masks_list: List[np.ndarray], 
                                     labels_list: List[List[str]], 
                                     confidences_list: List[np.ndarray],
                                     scene_info: Dict,
                                     save_path: Optional[str] = None,
                                     show_plot: bool = True) -> np.ndarray:
        """
        Visualize detection and segmentation results for both cameras side by side
        
        Args:
            images: List of BGR images [camera_0, camera_1]
            masks_list: List of instance masks for each camera
            labels_list: List of object labels for each camera
            confidences_list: List of detection confidences for each camera
            scene_info: Dictionary with scene information
            save_path: Path to save visualization
            show_plot: Whether to display the plot
            
        Returns:
            Visualization image
        """
        # Create figure with subplots for dual camera view
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Grounding SAM Results - {scene_info["scene"]} {scene_info["env_type"]}_{scene_info["env_id"]} Frame {scene_info["frame_id"]}', fontsize=16)
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for cam_idx, (image, masks, labels, confidences) in enumerate(zip(images, masks_list, labels_list, confidences_list)):
            # Convert BGR to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Original image
            axes[cam_idx, 0].imshow(image_rgb)
            axes[cam_idx, 0].set_title(f'Camera {cam_idx} - Original')
            axes[cam_idx, 0].axis('off')
            
            # All masks overlay
            axes[cam_idx, 1].imshow(image_rgb)
            
            detected_objects = []
            for i, (mask, label, conf) in enumerate(zip(masks, labels, confidences)):
                if i == 0:  # Skip background
                    continue
                color = colors[i % len(colors)]
                
                # Create colored mask
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask] = [*color[:3], 0.6]
                axes[cam_idx, 1].imshow(colored_mask)
                
                # Add label at center of mask
                y, x = np.where(mask)
                if len(y) > 0:
                    center_y, center_x = np.mean(y), np.mean(x)
                    axes[cam_idx, 1].text(center_x, center_y, f'{label}\n{conf:.2f}', 
                                         color='white', fontsize=8, ha='center', va='center',
                                         bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                    detected_objects.append(f'{label}({conf:.2f})')
            
            axes[cam_idx, 1].set_title(f'Camera {cam_idx} - All Detections')
            axes[cam_idx, 1].axis('off')
            
            # Individual masks (show up to 2 most confident objects)
            non_bg_indices = [(i, conf) for i, (label, conf) in enumerate(zip(labels, confidences)) if label != 'background']
            non_bg_indices.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
            
            for plot_idx in range(2):  # Show top 2 objects
                col_idx = plot_idx + 2
                if plot_idx < len(non_bg_indices):
                    mask_idx, conf = non_bg_indices[plot_idx]
                    mask = masks[mask_idx]
                    label = labels[mask_idx]
                    
                    axes[cam_idx, col_idx].imshow(image_rgb)
                    
                    # Overlay single mask
                    colored_mask = np.zeros((*mask.shape, 4))
                    colored_mask[mask] = [*colors[mask_idx % len(colors)][:3], 0.8]
                    axes[cam_idx, col_idx].imshow(colored_mask)
                    
                    axes[cam_idx, col_idx].set_title(f'{label} ({conf:.2f})')
                    axes[cam_idx, col_idx].axis('off')
                else:
                    axes[cam_idx, col_idx].axis('off')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dual camera visualization saved to: {save_path}")
        
        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        # Return the figure as an image array
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return vis_image

    def batch_process_environment(self, 
                                scene: str, 
                                env_type: str, 
                                env_id: int,
                                queries: Optional[List[str]] = None,
                                thresholds: Optional[List[float]] = None,
                                max_frames: int = 10,
                                output_dir: str = "./demo_results",
                                save_masks: bool = False) -> Dict:
        """
        Batch process an entire environment
        
        Args:
            scene: Scene name
            env_type: Environment type  
            env_id: Environment ID
            queries: Text queries (uses defaults if None)
            thresholds: Detection thresholds
            max_frames: Maximum number of frames to process
            output_dir: Output directory for results
            save_masks: Whether to save mask data
            
        Returns:
            Dictionary with processing results
        """
        if queries is None:
            queries = self.default_queries.get(scene, ['object', 'robotic arm'])
        
        if thresholds is None:
            thresholds = self.default_thresholds[:len(queries)]
            
        # Create output directory
        env_output_dir = os.path.join(output_dir, scene, f"{env_type}_{env_id}")
        os.makedirs(env_output_dir, exist_ok=True)
        
        results = {
            'scene': scene,
            'env_type': env_type,
            'env_id': env_id,
            'queries': queries,
            'processed_frames': [],
            'detection_stats': {}
        }
        
        # Process each camera
        for camera_id in [0, 1]:
            available_frames = self.get_available_frames(scene, env_type, env_id, camera_id)
            
            if not available_frames:
                print(f"No frames found for camera {camera_id}")
                continue
                
            frames_to_process = available_frames[:max_frames]
            
            print(f"Processing camera {camera_id}, {len(frames_to_process)} frames...")
            
            for frame_id in tqdm(frames_to_process, desc=f"Camera {camera_id}"):
                # Load image
                image = self.load_image(scene, env_type, env_id, camera_id, frame_id)
                if image is None:
                    continue
                
                # Process with Grounding SAM
                masks, labels, confidences = self.process_single_image(
                    image, queries, thresholds, 
                    cam_idx=camera_id, save_individual_masks=save_masks
                )
                
                # Save visualization
                vis_path = os.path.join(env_output_dir, f"camera_{camera_id}_frame_{frame_id}_vis.png")
                vis_image = self.visualize_results(
                    image, masks, labels, confidences, 
                    save_path=vis_path, show_plot=False
                )
                
                # Save mask data if requested
                if save_masks:
                    mask_data = {
                        'masks': masks.astype(bool).tolist(),
                        'labels': labels,
                        'confidences': confidences.tolist(),
                        'queries': queries,
                        'frame_info': {
                            'scene': scene,
                            'env_type': env_type,
                            'env_id': env_id,
                            'camera_id': camera_id,
                            'frame_id': frame_id
                        }
                    }
                    
                    mask_path = os.path.join(env_output_dir, f"camera_{camera_id}_frame_{frame_id}_masks.json")
                    with open(mask_path, 'w') as f:
                        json.dump(mask_data, f, indent=2)
                
                # Record results
                frame_result = {
                    'camera_id': camera_id,
                    'frame_id': frame_id,
                    'detected_objects': len([l for l in labels if l != 'background']),
                    'labels': labels,
                    'confidences': confidences.tolist()
                }
                results['processed_frames'].append(frame_result)
        
        # Compute detection statistics
        all_labels = []
        for frame in results['processed_frames']:
            all_labels.extend([l for l in frame['labels'] if l != 'background'])
        
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        results['detection_stats'] = dict(zip(unique_labels, counts.tolist()))
        
        # Save summary
        summary_path = os.path.join(env_output_dir, "processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Batch processing complete. Results saved to: {env_output_dir}")
        print(f"Detection statistics: {results['detection_stats']}")
        
        return results

    def demo_scene_overview(self, scene: str, max_envs: int = 3):
        """
        Create an overview of a scene by processing a few frames from different environments
        """
        print(f"\n=== Scene Overview: {scene} ===")
        
        environments = self.get_available_environments(scene)
        if not environments:
            print(f"No environments found for scene: {scene}")
            return
            
        print(f"Available environments: {environments}")
        
        # Process a few environments
        processed_count = 0
        for env_type, env_ids in environments.items():
            if processed_count >= max_envs:
                break
                
            for env_id in env_ids[:2]:  # Max 2 per environment type
                if processed_count >= max_envs:
                    break
                    
                print(f"\nProcessing {env_type}_{env_id}...")
                
                # Load and process a sample frame
                image = self.load_image(scene, env_type, env_id, camera_id=0, frame_id=0)
                if image is None:
                    continue
                    
                queries = self.default_queries.get(scene, ['object'])
                masks, labels, confidences = self.process_single_image(image, queries, [0.25])
                
                print(f"Detected: {[l for l in labels if l != 'background']}")
                
                # Show visualization
                self.visualize_results(image, masks, labels, confidences, show_plot=True)
                
                processed_count += 1

    def process_all_scenes_overview(self, 
                                  output_dir: str = "./all_scenes_demo",
                                  max_envs_per_scene: int = 2,
                                  frame_id: int = 0) -> Dict:
        """
        Process all available scenes and create overview visualizations
        
        Args:
            output_dir: Output directory for results
            max_envs_per_scene: Maximum environments to process per scene
            frame_id: Frame ID to process for each environment
            
        Returns:
            Dictionary with processing results
        """
        print("\n" + "="*60)
        print("Processing All Scenes Overview")
        print("="*60)
        
        all_results = {
            'processed_scenes': [],
            'total_detections': 0,
            'scene_summaries': {}
        }
        
        # Get all available scenes
        scenes = self.get_available_scenes()
        
        if not scenes:
            print("No scenes found in dataset")
            return all_results
            
        print(f"Found {len(scenes)} scenes: {scenes}")
        
        # Process each scene
        for scene in scenes:
            print(f"\n{'='*40}")
            print(f"Processing Scene: {scene}")
            print(f"{'='*40}")
            
            scene_output_dir = os.path.join(output_dir, scene)
            os.makedirs(scene_output_dir, exist_ok=True)
            
            environments = self.get_available_environments(scene)
            if not environments:
                print(f"No environments found for scene: {scene}")
                continue
                
            print(f"Available environments: {environments}")
            
            scene_results = {
                'scene': scene,
                'processed_envs': [],
                'detection_counts': {}
            }
            
            # Process a few environments per scene
            processed_count = 0
            for env_type, env_ids in environments.items():
                if processed_count >= max_envs_per_scene:
                    break
                    
                for env_id in env_ids[:1]:  # Process first environment of each type
                    if processed_count >= max_envs_per_scene:
                        break
                        
                    print(f"\nProcessing {env_type}_{env_id}...")
                    
                    # Load images from both cameras
                    image_cam0 = self.load_image(scene, env_type, env_id, camera_id=0, frame_id=frame_id)
                    image_cam1 = self.load_image(scene, env_type, env_id, camera_id=1, frame_id=frame_id)
                    
                    if image_cam0 is None or image_cam1 is None:
                        print(f"Could not load images for {env_type}_{env_id}")
                        continue
                    
                    # Get default queries for this scene
                    queries = self.default_queries.get(scene, ['object'])
                    thresholds = [0.25] * len(queries)
                    
                    print(f"Using queries: {queries}")
                    
                    # Process both cameras
                    masks_cam0, labels_cam0, confidences_cam0 = self.process_single_image(
                        image_cam0, queries, thresholds, cam_idx=0
                    )
                    
                    masks_cam1, labels_cam1, confidences_cam1 = self.process_single_image(
                        image_cam1, queries, thresholds, cam_idx=1
                    )
                    
                    # Create dual camera visualization
                    scene_info = {
                        'scene': scene,
                        'env_type': env_type,
                        'env_id': env_id,
                        'frame_id': frame_id
                    }
                    
                    vis_path = os.path.join(scene_output_dir, f"{env_type}_{env_id}_dual_camera.png")
                    
                    vis_image = self.visualize_dual_camera_results(
                        images=[image_cam0, image_cam1],
                        masks_list=[masks_cam0, masks_cam1],
                        labels_list=[labels_cam0, labels_cam1],
                        confidences_list=[confidences_cam0, confidences_cam1],
                        scene_info=scene_info,
                        save_path=vis_path,
                        show_plot=False
                    )
                    
                    # Count detections
                    detected_cam0 = [l for l in labels_cam0 if l != 'background']
                    detected_cam1 = [l for l in labels_cam1 if l != 'background']
                    
                    print(f"Camera 0 detected: {detected_cam0}")
                    print(f"Camera 1 detected: {detected_cam1}")
                    
                    # Record results
                    env_result = {
                        'env_type': env_type,
                        'env_id': env_id,
                        'camera_0_detections': detected_cam0,
                        'camera_1_detections': detected_cam1,
                        'total_objects': len(detected_cam0) + len(detected_cam1)
                    }
                    
                    scene_results['processed_envs'].append(env_result)
                    
                    # Update detection counts
                    for label in detected_cam0 + detected_cam1:
                        scene_results['detection_counts'][label] = scene_results['detection_counts'].get(label, 0) + 1
                    
                    all_results['total_detections'] += len(detected_cam0) + len(detected_cam1)
                    processed_count += 1
            
            # Save scene summary
            scene_summary_path = os.path.join(scene_output_dir, "scene_summary.json")
            with open(scene_summary_path, 'w') as f:
                json.dump(scene_results, f, indent=2)
            
            all_results['processed_scenes'].append(scene)
            all_results['scene_summaries'][scene] = scene_results['detection_counts']
            
            print(f"Scene {scene} complete. Detection counts: {scene_results['detection_counts']}")
        
        # Save overall summary
        summary_path = os.path.join(output_dir, "all_scenes_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        print(f"\n{'='*60}")
        print("All Scenes Processing Complete!")
        print(f"{'='*60}")
        print(f"Processed scenes: {all_results['processed_scenes']}")
        print(f"Total detections: {all_results['total_detections']}")
        print(f"Results saved to: {output_dir}")
        print(f"Scene summaries: {all_results['scene_summaries']}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Grounding SAM Demo for d3fields dataset')
    
    # Dataset parameters
    parser.add_argument('--dataset_root', type=str, 
                       default="/ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields",
                       help='Root directory of the adamanip_d3fields dataset')
    parser.add_argument('--scene', type=str, choices=['OpenBottle', 'OpenDoor', 'OpenCoffeeMachine', 
                                                     'OpenSafe', 'OpenWindow', 'OpenPressureCooker'],
                       help='Scene to process')
    parser.add_argument('--env_type', type=str, default='grasp_env',
                       help='Environment type (grasp_env, manip_env)')
    parser.add_argument('--env_id', type=int, default=0,
                       help='Environment ID')
    parser.add_argument('--camera_id', type=int, default=0, choices=[0, 1],
                       help='Camera ID')
    parser.add_argument('--frame_id', type=int, default=0,
                       help='Frame ID to process')
    
    # Processing parameters
    parser.add_argument('--queries', nargs='+', 
                       help='Text queries for object detection')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.25],
                       help='Detection thresholds')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='CUDA device')
    
    # Mode selection
    parser.add_argument('--list_scenes', action='store_true',
                       help='List available scenes and exit')
    parser.add_argument('--scene_overview', action='store_true',
                       help='Show overview of a scene')
    parser.add_argument('--all_scenes', action='store_true',
                       help='Process all available scenes with dual camera view')
    parser.add_argument('--dual_camera', action='store_true',
                       help='Show both cameras side by side (for single image mode)')
    parser.add_argument('--batch_process', action='store_true',
                       help='Batch process multiple frames')
    parser.add_argument('--max_frames', type=int, default=10,
                       help='Maximum frames to process in batch mode')
    parser.add_argument('--max_envs_per_scene', type=int, default=2,
                       help='Maximum environments to process per scene (for all_scenes mode)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./demo_results',
                       help='Output directory for results')
    parser.add_argument('--save_masks', action='store_true',
                       help='Save mask data as JSON files')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display visualizations')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = GroundingSAMDemo(device=args.device)
    demo.dataset_root = args.dataset_root
    
    # List scenes mode
    if args.list_scenes:
        scenes = demo.get_available_scenes()
        print("Available scenes:")
        for scene in scenes:
            environments = demo.get_available_environments(scene)
            print(f"  {scene}: {environments}")
        return
    
    # All scenes mode (new default behavior)
    if args.all_scenes or (not args.scene and not args.scene_overview and not args.batch_process):
        print("Processing all scenes with dual camera view...")
        results = demo.process_all_scenes_overview(
            output_dir=args.output_dir,
            max_envs_per_scene=args.max_envs_per_scene,
            frame_id=args.frame_id
        )
        return
    
    # Scene overview mode
    if args.scene_overview:
        if not args.scene:
            print("Please specify a scene with --scene")
            return
        demo.demo_scene_overview(args.scene)
        return
    
    # Batch processing mode
    if args.batch_process:
        if not args.scene:
            print("Please specify a scene with --scene")
            return
            
        results = demo.batch_process_environment(
            scene=args.scene,
            env_type=args.env_type,
            env_id=args.env_id,
            queries=args.queries,
            thresholds=args.thresholds,
            max_frames=args.max_frames,
            output_dir=args.output_dir,
            save_masks=args.save_masks
        )
        return
    
    # Single image processing mode
    if not args.scene:
        print("Please specify a scene with --scene, or use --list_scenes to see available options")
        return
        
    # Use default queries if not provided
    queries = args.queries or demo.default_queries.get(args.scene, ['object'])
    
    print(f"Processing image with queries: {queries}")
    print(f"Thresholds: {args.thresholds}")
    
    # Dual camera mode
    if args.dual_camera:
        # Load and process both cameras
        image_cam0 = demo.load_image(args.scene, args.env_type, args.env_id, 0, args.frame_id)
        image_cam1 = demo.load_image(args.scene, args.env_type, args.env_id, 1, args.frame_id)
        
        if image_cam0 is None or image_cam1 is None:
            print("Failed to load images from both cameras")
            return

        # Process both cameras
        masks_cam0, labels_cam0, confidences_cam0 = demo.process_single_image(image_cam0, queries, args.thresholds, cam_idx=0)
        masks_cam1, labels_cam1, confidences_cam1 = demo.process_single_image(image_cam1, queries, args.thresholds, cam_idx=1)

        print(f"Camera 0 detection results:")
        for i, (label, conf) in enumerate(zip(labels_cam0, confidences_cam0)):
            if label != 'background':
                print(f"  {label}: {conf:.3f}")

        print(f"Camera 1 detection results:")
        for i, (label, conf) in enumerate(zip(labels_cam1, confidences_cam1)):
            if label != 'background':
                print(f"  {label}: {conf:.3f}")

        # Create dual camera visualization
        scene_info = {
            'scene': args.scene,
            'env_type': args.env_type,
            'env_id': args.env_id,
            'frame_id': args.frame_id
        }

        save_path = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, 
                                   f"{args.scene}_{args.env_type}_{args.env_id}_frame{args.frame_id}_dual_camera.png")

        demo.visualize_dual_camera_results(
            images=[image_cam0, image_cam1],
            masks_list=[masks_cam0, masks_cam1],
            labels_list=[labels_cam0, labels_cam1],
            confidences_list=[confidences_cam0, confidences_cam1],
            scene_info=scene_info,
            save_path=save_path,
            show_plot=not args.no_display
        )
    else:
        # Single camera mode (original behavior)
        image = demo.load_image(args.scene, args.env_type, args.env_id, args.camera_id, args.frame_id)
        if image is None:
            print("Failed to load image")
            return

        # Process image
        masks, labels, confidences = demo.process_single_image(image, queries, args.thresholds)

        print(f"Detection results:")
        for i, (label, conf) in enumerate(zip(labels, confidences)):
            if label != 'background':
                print(f"  {label}: {conf:.3f}")

        # Visualize results
        save_path = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, 
                                   f"{args.scene}_{args.env_type}_{args.env_id}_cam{args.camera_id}_frame{args.frame_id}.png")

        demo.visualize_results(image, masks, labels, confidences, 
                              save_path=save_path, show_plot=not args.no_display)


if __name__ == "__main__":
    main() 