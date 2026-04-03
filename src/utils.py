import torch
import torch.nn as nn
import numpy as np
import re
import os
import random
import textwrap
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT


def save_checkpoint_to_drive(model, optimizer, epoch, loss, filename="autoencoder_checkpoint.pth"):
    """
    Saves the checkpoint directly to a specified folder in your mounted Google Drive.
    """
    # 1. Define the full Google Drive path
    # 'DL_Checkpoints' is the folder you want to save to inside your Drive
    drive_folder = '/content/gdrive/MyDrive/DL_Checkpoints'

    # Ensure the directory exists before attempting to save
    os.makedirs(drive_folder, exist_ok=True)

    # 2. Combine the folder and the filename
    full_path = os.path.join(drive_folder, filename)

    # 3. Create the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # 4. Save the dictionary to the Google Drive path
    torch.save(checkpoint, full_path)
    print(f"Checkpoint saved to Google Drive: {full_path} at epoch {epoch}")


def load_checkpoint_from_drive(model, optimizer=None, filename="autoencoder_checkpoint.pth"):
    """
    Loads a checkpoint from your Google Drive folder into the model and optimizer (if provided).
    """
    # Define the same Google Drive folder path
    drive_folder = '/content/gdrive/MyDrive/DL_Checkpoints'
    full_path = os.path.join(drive_folder, filename)

    # Check if the checkpoint file exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint file not found: {full_path}")

    # Load the checkpoint
    checkpoint = torch.load(full_path, map_location=torch.device('cpu'))  # use cuda if available

    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore optimizer state (if provided)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Extract metadata
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)

    print(f"Checkpoint loaded from: {full_path} (epoch {epoch})")

    return model, optimizer, epoch, loss


# This function just extracts the tags from the text, don't get distracted by it.
# I changed this function a bit to fix some bugs
def parse_gdi_text(text):
    """Parse GDI formatted text into structured data"""
    soup = BeautifulSoup(text, 'html.parser')
    images = []

    for gdi in soup.find_all('gdi'):
        # Debug: print what BeautifulSoup sees

        # Method 1: Try to get image attribute directly
        image_id = None
        if gdi.attrs:
            # Check for attributes like 'image1', 'image2', etc.
            for attr_name, attr_value in gdi.attrs.items():
                if 'image' in attr_name.lower():
                    image_id = attr_name.replace('image', '')
                    break

        # Method 2: Extract from the tag string using regex
        if not image_id:
            tag_str = str(gdi)
            match = re.search(r'<gdi\s+image(\d+)', tag_str)
            if match:
                image_id = match.group(1)

        # Method 3: Fallback - use sequential numbering
        if not image_id:
            image_id = str(len(images) + 1)

        content = gdi.get_text().strip()

        # Extract tagged elements using BeautifulSoup directly
        objects = [obj.get_text().strip() for obj in gdi.find_all('gdo')]
        actions = [act.get_text().strip() for act in gdi.find_all('gda')]
        locations = [loc.get_text().strip() for loc in gdi.find_all('gdl')]

        images.append({
            'image_id': image_id,
            'description': content,
            'objects': objects,
            'actions': actions,
            'locations': locations,
            'raw_text': str(gdi)
        })

    return images


# This is an utility function to show images.
# Why do we need to do all this?
def show_image(ax, image, de_normalize=False, img_mean=None, img_std=None):
    """
    De-normalize the image (if necessary) and show image
    """
    if de_normalize:
        new_mean = -img_mean / img_std
        new_std = 1 / img_std

        image = transforms.Normalize(
            mean=new_mean,
            std=new_std
        )(image)
    ax.imshow(image.permute(1, 2, 0))


def _parse_markdown_table(block: str) -> List[Dict[str, str]]:
    lines = [l.rstrip() for l in block.splitlines()]
    table_lines = [l for l in lines if l.strip().startswith("|")]
    if len(table_lines) < 3:
        return []
    header_line = table_lines[0]
    data_lines = table_lines[2:]
    headers = [h.strip() for h in header_line.strip("|").split("|")]
    rows = []
    for line in data_lines:
        if not line.strip().startswith("|"):
            break
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) != len(headers):
            continue
        rows.append(dict(zip(headers, cols)))
    return rows


def parse_cot_grounding(chain_of_thought: str) -> Dict[int, Dict[str, Any]]:
    """Parse StoryReasoning-style CoT markdown into per-frame bbox annotations."""
    frames: Dict[int, Dict[str, Any]] = {}
    img_pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(img_pattern.finditer(chain_of_thought or ""))

    for i, m in enumerate(matches):
        img_idx = int(m.group(1)) - 1
        start = m.end()
        end = matches[i + 1].start() if (i + 1 < len(matches)) else len(chain_of_thought)
        section = (chain_of_thought or "")[start:end]

        frames[img_idx] = {"characters": [], "objects": []}

        char_match = re.search(r"###\s*Characters(.*?)(?=\n###|\n##|$)", section, flags=re.DOTALL)
        if char_match:
            for row in _parse_markdown_table(char_match.group(1)):
                cid = row.get("Character ID", "").strip()
                bbox_str = row.get("Bounding Box", "").strip()
                if cid and bbox_str:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox_str.split(",")]
                        frames[img_idx]["characters"].append({"id": cid, "bbox": [x1, y1, x2, y2]})
                    except Exception:
                        pass

        obj_match = re.search(r"###\s*Objects(.*?)(?=\n###|\n##|$)", section, flags=re.DOTALL)
        if obj_match:
            for row in _parse_markdown_table(obj_match.group(1)):
                oid = row.get("Object ID", "").strip()
                bbox_str = row.get("Bounding Box", "").strip()
                if oid and bbox_str:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox_str.split(",")]
                        frames[img_idx]["objects"].append({"id": oid, "bbox": [x1, y1, x2, y2]})
                    except Exception:
                        pass
    return frames


def _clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def crop_and_resize(pil_img, bbox, out_hw=(60, 125)):
    x1, y1, x2, y2 = bbox
    W, H = pil_img.size
    x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, W, H)
    crop = pil_img.crop((x1, y1, x2, y2))
    crop = transforms.Resize(out_hw)(crop)
    crop = transforms.ToTensor()(crop)
    return crop


def pick_reid_pair(frames_cot: Dict[int, Dict[str, Any]]) -> Optional[Tuple[int, int, List[int], List[int], str]]:
    """Pick two detections of the same entity id across frames."""
    id_to_dets = {}
    for f_idx, content in frames_cot.items():
        for det in content.get("characters", []) + content.get("objects", []):
            ent_id = det.get("id")
            bbox = det.get("bbox")
            if ent_id and bbox:
                id_to_dets.setdefault(ent_id, []).append((f_idx, bbox))

    candidates = [ent_id for ent_id, dets in id_to_dets.items() if len(dets) >= 2]
    if not candidates:
        return None

    ent_id = random.choice(candidates)
    dets = id_to_dets[ent_id]
    (f1, b1), (f2, b2) = random.sample(dets, 2)
    return f1, f2, b1, b2, ent_id


def extract_cot_text_for_frame(chain_of_thought: str, frame_idx: int, max_chars: int = 600) -> str:
    """Option 4 helper: extract the 'Image N' section as plain text (best-effort)."""
    if not chain_of_thought:
        return ""
    img_pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(img_pattern.finditer(chain_of_thought))
    target = None
    for i, m in enumerate(matches):
        if int(m.group(1)) - 1 == frame_idx:
            start = m.end()
            end = matches[i + 1].start() if (i + 1 < len(matches)) else len(chain_of_thought)
            target = chain_of_thought[start:end]
            break
    if target is None:
        return ""
    # Remove markdown tables (keep only non-table lines)
    lines = []
    for line in target.splitlines():
        if line.strip().startswith("|"):
            continue
        if set(line.strip()) <= set("-|:"):
            continue
        lines.append(line)
    text = " ".join([l.strip() for l in lines if l.strip()])
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


# Turn these on/off to control the 4 optional improvements.
USE_FRAME_AWARE_GROUNDING = True  # Option 2: align ROI to matching frame text embedding (instead of always frame 0)
USE_CONTRASTIVE_ROI = True  # Option 1: InfoNCE-style contrastive grounding using batch negatives
USE_ENTITY_POOLING = True  # Option 3: entity-specific pooling/consistency across batch by entity_id
USE_COT_TEXT = True  # Option 4: concatenate CoT text snippet to the frame descriptions

# Contrastive temperature (only used if USE_CONTRASTIVE_ROI=True)
CONTRASTIVE_TAU = 0.07

# @title Main dataset
"""
Defines the `SequencePredictionDataset` class, which is the core data provider for the main task.
1. `__getitem__`:
   - Loads 5 frames (4 context + 1 target).
   - Parses text descriptions and optionally appends CoT text.
   - Extracts bounding box crops (ROIs) for grounding tasks if CoT data is available.
   - Returns a tuple containing: sequence images, descriptions, target image, target text, ROI crops, and validity flags.
"""


class SequencePredictionDataset(Dataset):
    def __init__(self, original_dataset, tokenizer, K: int = 4, max_len: int = 120, image_hw=(60, 125)):
        super(SequencePredictionDataset, self).__init__()
        self.dataset = original_dataset
        self.tokenizer = tokenizer
        self.K = K
        self.max_len = max_len
        self.image_hw = image_hw

        # Potential experiments: Try other transforms!
        self.transform = transforms.Compose([
            transforms.Resize(image_hw),  # Reasonable size based on our previous analysis
            transforms.ToTensor(),  # HxWxC -> CxHxW
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Selects a 5 frame sequence from the dataset. Sets 4 for training and the last one
        as a target.

        Returns:
          frames:        [K, C, H, W]
          descriptions:  [K, T]
          image_target:  [C, H, W]
          target_ids:    [1, T]
          roi1, roi2:    [C, H, W] (cropped from CoT bboxes, if available)
          roi_valid:     0/1
          roi_frame:     frame index for roi1 (0..K-1) if available else -1
          ent_id:        string id for the ROI entity (empty if none)
        """
        frames = self.dataset[idx]["images"]
        image_attributes = parse_gdi_text(self.dataset[idx]["story"])

        # CoT grounding annotations (may be missing / unparseable)
        cot = self.dataset[idx].get("chain_of_thought", "")
        cot_frames = parse_cot_grounding(cot)

        frame_tensors = []
        description_list = []

        for frame_idx in range(self.K):
            image = FT.equalize(frames[frame_idx])
            input_frame = self.transform(image)
            frame_tensors.append(input_frame)

            description = image_attributes[frame_idx]["description"]

            # Option 4: include CoT text snippet for this frame (best-effort)
            if USE_COT_TEXT:
                cot_txt = extract_cot_text_for_frame(cot, frame_idx)
                if cot_txt:
                    description = description + " [COT] " + cot_txt

            input_ids = self.tokenizer(
                description,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_len
            ).input_ids.squeeze(0)

            description_list.append(input_ids)

        image_target = FT.equalize(frames[self.K])
        image_target = self.transform(image_target)

        target_desc = image_attributes[self.K]["description"]
        target_ids = self.tokenizer(
            target_desc,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        ).input_ids  # [1, T]

        # ---- CoT ROI pair (Options 1-3 need these) ----
        roi_valid = torch.tensor(0, dtype=torch.long)
        roi1 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi2 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi_frame = torch.tensor(-1, dtype=torch.long)
        ent_id = ""

        pair = pick_reid_pair(cot_frames)
        if pair is not None:
            f1, f2, b1, b2, ent_id = pair
            # We only use ROIs that fall within the input window (0..K-1)
            if (0 <= f1 < self.K) and (0 <= f2 < self.K):
                try:
                    roi1 = crop_and_resize(frames[f1], b1, out_hw=self.image_hw)
                    roi2 = crop_and_resize(frames[f2], b2, out_hw=self.image_hw)
                    roi_valid = torch.tensor(1, dtype=torch.long)
                    roi_frame = torch.tensor(int(f1), dtype=torch.long)
                except Exception:
                    pass

        sequence_tensor = torch.stack(frame_tensors)  # [K, C, H, W]
        description_tensor = torch.stack(description_list)  # [K, T]

        return (
            sequence_tensor,
            description_tensor,
            image_target,
            target_ids,
            roi1, roi2, roi_valid, roi_frame, ent_id
        )


"""We will use text autoencoding (reconstructing the same text) to develop representations of the text (I provide some existing weights for this, but you can train your own)"""

# @title Text task dataset (text autoencoding)
"""
Defines `TextTaskDataset` for pre-training or fine-tuning the text encoder separately.
It simply pulls a random text description from a story to perform text-to-text autoencoding.
"""


class TextTaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        num_frames = self.dataset[idx]["frame_count"]
        self.image_attributes = parse_gdi_text(self.dataset[idx]["story"])

        # Pick
        frame_idx = np.random.randint(0, 5)
        description = self.image_attributes[frame_idx]["description"]

        return description  # Returning the whole description


"""And also a dataset for a potential image autoencoder task if you want to develop some visual features before training the whose archicture."""

# @title Dataset for image autoencoder task
"""
Defines `AutoEncoderTaskDataset` for pre-training the visual autoencoder.
It retrieves a single random frame from the dataset to learn image reconstruction (Image -> Latent -> Image).
"""


class AutoEncoderTaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize((240, 500)),  # Reasonable size based on our previous analysis
            transforms.ToTensor(),  # HxWxC -> CxHxW
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        num_frames = self.dataset[idx]["frame_count"]
        frames = self.dataset[idx]["images"]

        # Pick a frame at random
        frame_idx = torch.randint(0, 5, (1,)).item()
        input_frame = self.transform(frames[frame_idx])  # Input to the autoencoder

        return input_frame,  # Returning the image


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)


# Plots four images and their reconstructions
def validation(model, val_dataloader, tokenizer, device):
    model.eval()
    with torch.no_grad():
        # Unpack 9 values (dataset was updated to return CoT info)
        frames, descriptions, image_target, text_target, roi1, roi2, roi_valid, roi_frame, ent_id = next(
            iter(val_dataloader))

        descriptions = descriptions.to(device)
        frames = frames.to(device)
        image_target = image_target.to(device)
        text_target = text_target.to(device)

        # Unpack 7 values (model now returns extra latents for grounding)
        predicted_image_k, context_image, _, hidden, cell, _, _ = model(frames, descriptions, text_target)

        figure, ax = plt.subplots(2, 6, figsize=(20, 5), gridspec_kw={'height_ratios': [2, 1.5]})

        for i in range(4):
            im = frames[0, i, :, :, :].cpu()
            show_image(ax[0, i], im)
            ax[0, i].set_aspect('auto')
            ax[0, i].axis('off')
            wrapped_text = textwrap.fill(tokenizer.decode(descriptions[0, i, :], skip_special_tokens=True), width=40)

            ax[1, i].text(
                0.5, 0.99,
                wrapped_text,
                ha='center',
                va='top',
                fontsize=10,
                wrap=True
            )

            ax[1, i].axis('off')  # Hide axes for the text subplot

        show_image(ax[0, 4], image_target[0].cpu())
        ax[0, 4].set_title('Target')
        ax[0, 4].set_aspect('auto')
        ax[0, 4].axis('off')
        text_target = text_target.squeeze(1)

        wrapped_text = textwrap.fill(tokenizer.decode(text_target[0], skip_special_tokens=True), width=40)
        ax[1, 4].text(
            0.5, 0.99,
            wrapped_text,
            ha='center',
            va='top',
            fontsize=10,
            wrap=False)
        ax[1, 4].axis('off')
        output = context_image[0, :, :, :].cpu()
        show_image(ax[0, 5], output)
        ax[0, 5].set_title('Predicted')
        ax[0, 5].set_aspect('auto')
        ax[0, 5].axis('off')

        generated_tokens = generate(model.text_decoder,
                                    hidden[:, 0, :].unsqueeze(1),
                                    cell[:, 0, :].unsqueeze(1),
                                    max_len=150,
                                    sos_token_id=tokenizer.cls_token_id,
                                    eos_token_id=tokenizer.sep_token_id, device=device)

        wrapped_text = textwrap.fill(tokenizer.decode(generated_tokens), width=40)

        ax[1, 5].text(
            0.5, 0.99,
            wrapped_text,
            ha='center',
            va='top',
            fontsize=10,
            wrap=False)
        ax[1, 5].axis('off')
        plt.tight_layout()
        plt.show()


def generate(model, hidden, cell, max_len, sos_token_id, eos_token_id, device):
    """
      This function generates a sequence of tokens using the provided decoder.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # 2. SETUP DECODER INPUT
    # Start with the SOS token, shape (1, 1)
    dec_input = torch.tensor([[sos_token_id]], dtype=torch.long, device=device)
    # hidden = torch.zeros(1, 1, hidden_dim, device=device)
    # cell = torch.zeros(1, 1, hidden_dim, device=device)

    generated_tokens = []

    # 3. AUTOREGRESSIVE LOOP
    for _ in range(max_len):
        with torch.no_grad():
            # Run the decoder one step at a time
            # dec_input is (1, 1) here—it's just the last predicted token
            prediction, hidden, cell = model(dec_input, hidden, cell)

        logits = prediction.squeeze(1)  # Shape (1, vocab_size)
        temperature = 0.9  # <--- Try a value between 0.5 and 1.0

        # 1. Divide logits by temperature
        # 2. Apply softmax to get probabilities
        # 3. Use multinomial to sample one token based on the probabilities
        probabilities = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        token_id = next_token.squeeze().item()

        # Check for the End-of-Sequence token
        if token_id == eos_token_id:
            break

        if token_id == 0 or token_id == sos_token_id:
            continue

        # Append the predicted token
        generated_tokens.append(token_id)

        # The predicted token becomes the input for the next iteration
        dec_input = next_token

    # Return the list of generated token IDs
    return generated_tokens
