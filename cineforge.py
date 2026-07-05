import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

# =====================================================================
# 1. ADVANCED CINEMATIC DEEP LAYER ARCHITECTURE (CineNet v1.0 )
# =====================================================================
class AdvancedCineNet(nn.Module):
    """
    High-performance, CPU-friendly feature extraction network mimicking
    DaVinci Resolve's 3D LUT lattice interpolations and color transform matrices.
    """
    def __init__(self):
        super(AdvancedCineNet, self).__init__()
        # Deep feature extraction for micro-contrast & tone response
        self.input_layer = nn.Conv2d(3, 32, kernel_size=1)
        
        self.residual_block = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            nn.Mish(),  # Advanced non-linear activation for smoother grading transitions
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Mish(),
            nn.Conv2d(64, 32, kernel_size=1)
        )
        
        # Cross-channel interaction simulating matrix saturation configurations
        self.chroma_tuning = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh()  # Safeguards strict cinematic dynamic range boundaries
        )

    def forward(self, x):
        identity = self.input_layer(x)
        res = self.residual_block(identity)
        out = self.chroma_tuning(identity + res)
        return out

# =====================================================================
# 2. ADVANCED CINEMATOGRAPHY COMPONENT MOTOR (Sony FX & DaVinci Simulation)
# =====================================================================
class CineEngine:
    @staticmethod
    def apply_sony_fx_grain(img, intensity=0.014):
        """Simulates native Sony FX Cine-Alta sensor noise at structural grain levels."""
        grain = np.random.normal(0, intensity, img.shape).astype(np.float32)
        # Luminance masking: grain is mostly visible in shadows and midtones, clean in highlights
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        luma_mask = np.expand_dims(1.0 - (gray ** 2), axis=2)
        return np.clip(img + (grain * luma_mask), 0, 1)

    @staticmethod
    def simulate_halation(img, threshold=0.82, blur_radius=51, intensity=0.25):
        """Replicates film halation (red glow around high-exposure edge transitions)."""
        bright_pixels = np.clip(img - threshold, 0, 1) * (1.0 / (1.0 - threshold))
        halation_layer = np.zeros_like(img)
        halation_layer[:, :, 0] = bright_pixels[:, :, 0] * intensity  # Primary Red channel bleeding
        halation_layer[:, :, 1] = bright_pixels[:, :, 1] * (intensity * 0.15)
        
        halation_blur = cv2.GaussianBlur(halation_layer, (blur_radius, blur_radius), 0)
        return np.clip(img + halation_blur, 0, 1)

    @staticmethod
    def anamorphic_flare_sim(img, threshold=0.92, intensity=0.4):
        """Simulates horizontal anamorphic lens streak light leak dynamics across intense points."""
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
        
        flare_mask = np.zeros_like(img)
        # Generate structural horizontal streaking using a highly directional kernel
        kernel = np.zeros((3, 91), dtype=np.uint8)
        kernel[1, :] = 1
        
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        # Apply cinematic blue tint to the streak flares
        flare_mask[:, :, 0] = dilated_mask * (intensity * 0.2)  # Red
        flare_mask[:, :, 1] = dilated_mask * (intensity * 0.5)  # Green
        flare_mask[:, :, 2] = dilated_mask * intensity          # Blue (Cinematic Anamorphic Blue)
        
        flare_blur = cv2.GaussianBlur(flare_mask.astype(np.float32) / 255.0, (41, 11), 0)
        return np.clip(img + flare_blur, 0, 1)

    @staticmethod
    def professional_vignette(img, strength=0.35):
        """Applies a subtle, optical-vignette falloff to pull viewer focus to the center."""
        h, w, _ = img.shape
        kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (w, 1))
        kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h))
        
        # Calculate radial distances
        vignette_x = cv2.GaussianBlur(kernel_x.astype(np.float32), (w | 1, 1), w / 2)
        vignette_y = cv2.GaussianBlur(kernel_y.astype(np.float32), (1, h | 1), h / 2)
        vignette = np.dot(vignette_y.T, vignette_x)
        
        vignette = vignette / np.max(vignette)
        vignette = cv2.resize(vignette, (w, h))
        vignette = np.expand_dims(vignette, axis=2)
        
        # Map strength interpolation
        vignette = (1.0 - strength) + (vignette * strength)
        return np.clip(img * vignette, 0, 1)

    @staticmethod
    def davinci_tone_mapping(img):
        """Applies an elegant film S-Curve response preventing highlights from clipping."""
        # Non-linear log mapping curve
        res = 1.0 / (1.0 + np.exp(-9.5 * (img - 0.5)))
        return cv2.addWeighted(img, 0.25, res, 0.75, 0)

    @staticmethod
    def bleach_bypass_emulation(img, amount=0.2):
        """Simulates structural silver-retention film processing for high micro-contrast."""
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray_3ch = np.stack([gray, gray, gray], axis=-1)
        
        mask = img < 0.5
        blend = np.zeros_like(img)
        blend[mask] = 2 * img[mask] * gray_3ch[mask]
        blend[~mask] = 1 - 2 * (1 - img[~mask]) * (1 - gray_3ch[~mask])
        return np.clip((1 - amount) * img + amount * blend, 0, 1)

# =====================================================================
# 3. SELF-SUPERVISED MACHINE LEARNING LOOP
# =====================================================================
def self_train_on_image(img_np):
    """
    Executes a zero-dataset, hardware-friendly on-the-fly training loop
    optimizing the neural network specifically to target the calculated scene characteristics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mathematical Target Formulation via ISP Pipelines
    target = CineEngine.davinci_tone_mapping(img_np)
    target = CineEngine.bleach_bypass_emulation(target, amount=0.12)
    
    # Input Feature Transformation Formulation
    src_input = CineEngine.simulate_halation(img_np)
    
    # Resize down to an optimized 384x384 matrix for instant CPU training optimization
    src_res = cv2.resize(src_input, (384, 384))
    tgt_res = cv2.resize(target, (384, 384))
    
    in_tensor = torch.from_numpy(src_res).permute(2, 0, 1).unsqueeze(0).to(device)
    out_tensor = torch.from_numpy(tgt_res).permute(2, 0, 1).unsqueeze(0).to(device)
    
    model = AdvancedCineNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.006, weight_decay=1e-4)
    criterion = nn.HuberLoss() # Highly stable loss function resilient against saturation gradients
    
    model.train()
    for _ in range(90):
        optimizer.zero_grad()
        pred = (model(in_tensor) + 1) / 2.0
        loss = criterion(pred, out_tensor)
        loss.backward()
        optimizer.step()
        
    return model

# =====================================================================
# 4. ZERO-CONFIGURATION EXECUTION ENGINE
# =====================================================================
def main():
    print("\n" + "="*60)
    print("      CINEFORGE CORE v1.0 : AUTOMATED PIPELINE RUNNING ")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Search working folder for target assets
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(current_dir, ext)))
        
    # Guard against pipeline recursion loops
    image_files = [f for f in image_files if not os.path.basename(f).startswith("cineforge_")]

    if not image_files:
        print("[-] Error: No source images discovered in the directory.")
        print("[-] Instruction: Place .jpg or .png graphics next to this script and rerun.")
        return

    print(f"[+] Total target media identified: {len(image_files)} image(s).")
    print(f"[+] Compute Hardware Target: {device.type.upper()}")
    print("[*] Instantiating neural color matching pipelines...\n")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"[>] Processing: {filename}")
        
        # Ingest asset in true dynamic range space
        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            print(f" [!] Warning: Failed to parse matrix for {filename}. Skipping.")
            continue
            
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        img_float = orig_rgb.astype(np.float32) / 255.0
        
        # Step 1: Execute Instant Self-Supervised Neural Optimization
        model = self_train_on_image(img_float)
        
        # Step 2: Native Resolution Inference
        model.eval()
        in_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_tensor = (model(in_tensor) + 1) / 2.0
            pred_tensor = torch.clamp(pred_tensor, 0, 1)
            
        output_np = pred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Step 3: Secondary Advanced Optics & Post-Production FX passes
        print("     └─ Simulating Anamorphic Optics, Halation and Cine-Alta Film Grain...")
        output_np = CineEngine.anamorphic_flare_sim(output_np, intensity=0.15)
        output_np = CineEngine.simulate_halation(output_np, intensity=0.12)
        output_np = CineEngine.professional_vignette(output_np, strength=0.25)
        output_np = CineEngine.apply_sony_fx_grain(output_np, intensity=0.008)
        
        # Write matrix payload back to local storage
        output_bgr = cv2.cvtColor((output_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_path = os.path.join(current_dir, f"cineforge_{filename}")
        cv2.imwrite(output_path, output_bgr)
        print(f"[✓] Successfully forged: cineforge_{filename}")

    print("\n" + "="*60)
    print(" Pipeline Complete. Source media upgraded to cinematic master spec!")
    print("="*60)

if __name__ == "__main__":
    main()