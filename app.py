"""
Yoga AI Pose Detection and Correction Application
Main Tkinter GUI application with image upload
"""
# cd "C:\Users\isha choudhary\OneDrive\Desktop\minor project\yoga_ai"
# py -3.11 app.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

from pose_detector import PoseDetector
from angle_utils import calculate_all_angles
from feedback_engine import FeedbackEngine, recognize_pose
from skeleton_drawer import SkeletonDrawer
from pose_rules import get_all_pose_names


class YogaAIApp:
    def __init__(self, root):
        """Initialize the main application"""
        self.root = root
        self.root.title("Yoga AI - Pose Detection & Correction")
        self.root.geometry("1200x900")
        self.root.configure(bg='#2c3e50')
        self.root.minsize(1000, 700)
        
        # Initialize components
        self.pose_detector = PoseDetector(smoothing_window=5)
        self.feedback_engine = FeedbackEngine()
        self.skeleton_drawer = SkeletonDrawer()
        
        # Image tracking
        self.current_image = None
        self.current_image_path = None
        
        # Pose tracking
        self.current_pose = "Unknown"
        self.current_angles = {}
        self.current_landmarks = {}
        self.confidence = 0.0
        
        # Setup UI
        self.setup_ui()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Image display and overlay
        left_panel = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image label
        image_label = tk.Label(left_panel, text="Upload Your Yoga Pose Photo", 
                               bg='#34495e', fg='white', 
                               font=('Arial', 14, 'bold'))
        image_label.pack(pady=10)
        
        # Control buttons - Put them at the TOP for visibility
        button_frame = tk.Frame(left_panel, bg='#34495e')
        button_frame.pack(pady=15, padx=10, fill=tk.X)
        
        self.upload_button = tk.Button(button_frame, text="📷 Upload Photo", 
                                      command=self.upload_image,
                                      bg='#3498db', fg='white',
                                      font=('Arial', 14, 'bold'),
                                      width=18, height=2,
                                      cursor='hand2')
        self.upload_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.analyze_button = tk.Button(button_frame, text="🔍 Analyze Pose", 
                                       command=self.analyze_pose,
                                       bg='#27ae60', fg='white',
                                       font=('Arial', 14, 'bold'),
                                       width=18, height=2,
                                       state=tk.DISABLED,
                                       cursor='hand2')
        self.analyze_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.clear_button = tk.Button(button_frame, text="🗑️ Clear", 
                                     command=self.clear_image,
                                     bg='#e74c3c', fg='white',
                                     font=('Arial', 14, 'bold'),
                                     width=12, height=2,
                                     state=tk.DISABLED,
                                     cursor='hand2')
        self.clear_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Image display
        self.image_label = tk.Label(left_panel, bg='#1a1a1a', 
                                   width=640, height=480,
                                   text="No image loaded\n\nClick '📷 Upload Photo' button above to begin",
                                   fg='white', font=('Arial', 12))
        self.image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Right panel - Information and feedback
        right_panel = tk.Frame(main_frame, bg='#34495e', width=400,
                              relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Title
        title_label = tk.Label(right_panel, text="Yoga AI Assistant", 
                              bg='#34495e', fg='white',
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=15)
        
        # Confidence display
        confidence_frame = tk.Frame(right_panel, bg='#34495e')
        confidence_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(confidence_frame, text="Confidence:", 
                bg='#34495e', fg='white', font=('Arial', 11)).pack(side=tk.LEFT)
        self.confidence_label = tk.Label(confidence_frame, text="0.0%", 
                                        bg='#34495e', fg='#f39c12',
                                        font=('Arial', 11, 'bold'))
        self.confidence_label.pack(side=tk.LEFT, padx=10)
        
        # Pose selection
        pose_frame = tk.Frame(right_panel, bg='#34495e')
        pose_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(pose_frame, text="Select Pose:", 
                bg='#34495e', fg='white', font=('Arial', 11)).pack(anchor=tk.W)
        
        self.pose_var = tk.StringVar(value="Auto-Detect")
        pose_combo = ttk.Combobox(pose_frame, textvariable=self.pose_var,
                                 values=["Auto-Detect"] + get_all_pose_names(),
                                 state='readonly', width=25)
        pose_combo.pack(pady=5, fill=tk.X)
        pose_combo.bind('<<ComboboxSelected>>', self.on_pose_selected)
        
        # Detected pose
        detected_frame = tk.Frame(right_panel, bg='#34495e')
        detected_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(detected_frame, text="Detected Pose:", 
                bg='#34495e', fg='white', font=('Arial', 11)).pack(anchor=tk.W)
        self.detected_pose_label = tk.Label(detected_frame, text="Unknown", 
                                           bg='#34495e', fg='#3498db',
                                           font=('Arial', 12, 'bold'))
        self.detected_pose_label.pack(pady=5, anchor=tk.W)
        
        # Angles display
        angles_frame = tk.LabelFrame(right_panel, text="Joint Angles", 
                                    bg='#34495e', fg='white',
                                    font=('Arial', 11, 'bold'))
        angles_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.angles_text = tk.Text(angles_frame, bg='#2c3e50', fg='#ecf0f1',
                                  font=('Courier', 10), height=8,
                                  wrap=tk.WORD, state=tk.DISABLED)
        self.angles_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Feedback panel
        feedback_frame = tk.LabelFrame(right_panel, text="Correction Feedback", 
                                       bg='#34495e', fg='white',
                                       font=('Arial', 11, 'bold'))
        feedback_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Scrollbar for feedback
        feedback_scroll = tk.Scrollbar(feedback_frame)
        feedback_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.feedback_text = tk.Text(feedback_frame, bg='#2c3e50', fg='#e74c3c',
                                     font=('Arial', 11), height=6,
                                     wrap=tk.WORD, state=tk.DISABLED,
                                     yscrollcommand=feedback_scroll.set)
        self.feedback_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        feedback_scroll.config(command=self.feedback_text.yview)
    
    def upload_image(self):
        """Upload and display an image"""
        file_path = filedialog.askopenfilename(
            title="Select a Yoga Pose Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Read image using OpenCV
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Could not load image. Please try a different file.")
                return
            
            self.current_image = image
            self.current_image_path = file_path
            
            # Display the image
            self.display_image(image)
            
            # Enable analyze and clear buttons
            self.analyze_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL)
            
            # Reset pose data
            self.current_pose = "Unknown"
            self.current_angles = {}
            self.current_landmarks = {}
            self.confidence = 0.0
            self.detected_pose_label.config(text="Unknown")
            self.update_ui(0.0, {}, [])
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image):
        """Display image in the label"""
        try:
            # Resize image to fit the label if needed
            height, width = image.shape[:2]
            max_width, max_height = 640, 480
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_resized = cv2.resize(image, (new_width, new_height))
            else:
                image_resized = image
            
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image=image_pil)
            
            self.image_label.config(image=image_tk, text="")
            self.image_label.image = image_tk  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def analyze_pose(self):
        """Analyze the uploaded image for pose detection"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        
        try:
            # Create a copy of the image for processing
            frame = self.current_image.copy()
            
            # Detect pose
            landmarks, confidence, mp_landmarks = self.pose_detector.detect(frame)
            self.confidence = confidence
            
            if not landmarks:
                messagebox.showinfo("No Pose Detected", 
                                  "Could not detect a pose in the image.\n\n"
                                  "Please ensure:\n"
                                  "• The person is clearly visible\n"
                                  "• The image shows a full body view\n"
                                  "• There is good lighting")
                return
            
            # Calculate angles
            self.current_landmarks = landmarks
            self.current_angles = calculate_all_angles(landmarks)
            
            # Recognize pose if auto-detect is selected
            if self.pose_var.get() == "Auto-Detect":
                self.current_pose = recognize_pose(self.current_angles, landmarks)
                self.detected_pose_label.config(text=self.current_pose)
            else:
                self.current_pose = self.pose_var.get()
                self.detected_pose_label.config(text=self.current_pose)
            
            # Draw skeleton on the image
            frame = self.skeleton_drawer.draw_skeleton(frame, landmarks)
            
            # Draw angles
            frame = self.skeleton_drawer.draw_angles(frame, self.current_angles, landmarks)
            
            # Generate feedback
            feedback_messages = self.feedback_engine.analyze_pose(
                landmarks, self.current_angles, self.current_pose
            )
            
            # Draw confidence and pose name
            frame = self.skeleton_drawer.draw_confidence(frame, confidence)
            frame = self.skeleton_drawer.draw_pose_name(frame, self.current_pose)
            
            # Display the annotated image
            self.display_image(frame)
            
            # Update UI
            self.update_ui(confidence, self.current_angles, feedback_messages)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze pose: {str(e)}")
    
    def clear_image(self):
        """Clear the current image and reset"""
        self.current_image = None
        self.current_image_path = None
        self.current_pose = "Unknown"
        self.current_angles = {}
        self.current_landmarks = {}
        self.confidence = 0.0
        
        self.image_label.config(image='', 
                               text="No image loaded\nClick 'Upload Photo' to begin")
        self.image_label.image = None
        
        self.analyze_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        
        self.detected_pose_label.config(text="Unknown")
        self.update_ui(0.0, {}, [])
    
    def on_pose_selected(self, event=None):
        """Handle pose selection change"""
        selected = self.pose_var.get()
        if selected != "Auto-Detect":
            self.current_pose = selected
            self.detected_pose_label.config(text=selected)
            
            # Re-analyze if image is loaded
            if self.current_image is not None:
                self.analyze_pose()
    
    def update_ui(self, confidence, angles, feedback_messages):
        """Update UI elements with current data"""
        # Update confidence
        conf_percent = confidence * 100
        color = '#27ae60' if conf_percent > 70 else '#f39c12' if conf_percent > 50 else '#e74c3c'
        self.confidence_label.config(text=f"{conf_percent:.1f}%", fg=color)
        
        # Update angles display
        self.angles_text.config(state=tk.NORMAL)
        self.angles_text.delete(1.0, tk.END)
        
        angles_str = ""
        if angles.get('knee'):
            knee = angles['knee']
            angles_str += f"Knee:\n"
            if knee.get('left'):
                angles_str += f"  Left: {knee['left']:.1f}°\n"
            if knee.get('right'):
                angles_str += f"  Right: {knee['right']:.1f}°\n"
        
        if angles.get('elbow'):
            elbow = angles['elbow']
            angles_str += f"\nElbow:\n"
            if elbow.get('left'):
                angles_str += f"  Left: {elbow['left']:.1f}°\n"
            if elbow.get('right'):
                angles_str += f"  Right: {elbow['right']:.1f}°\n"
        
        if angles.get('shoulder'):
            shoulder = angles['shoulder']
            angles_str += f"\nShoulder:\n"
            if shoulder.get('left'):
                angles_str += f"  Left: {shoulder['left']:.1f}°\n"
            if shoulder.get('right'):
                angles_str += f"  Right: {shoulder['right']:.1f}°\n"
        
        if angles.get('hip'):
            hip = angles['hip']
            angles_str += f"\nHip:\n"
            if hip.get('left'):
                angles_str += f"  Left: {hip['left']:.1f}°\n"
            if hip.get('right'):
                angles_str += f"  Right: {hip['right']:.1f}°\n"
        
        if angles.get('spine') is not None:
            angles_str += f"\nSpine: {angles['spine']:.1f}°\n"
        
        if not angles_str:
            angles_str = "No angle data available"
        
        self.angles_text.insert(1.0, angles_str)
        self.angles_text.config(state=tk.DISABLED)
        
        # Update feedback
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.delete(1.0, tk.END)
        
        if feedback_messages:
            feedback_str = "\n".join([f"• {msg}" for msg in feedback_messages])
            self.feedback_text.insert(1.0, feedback_str)
        else:
            self.feedback_text.insert(1.0, "No feedback available. Upload an image and analyze to get feedback.")
        
        self.feedback_text.config(state=tk.DISABLED)
    
    def on_closing(self):
        """Handle window closing"""
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = YogaAIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
