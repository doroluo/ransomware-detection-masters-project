import os
import random
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def print_classification_report(y_true, y_pred, class_names):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    print(f"{'class':<22} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for i, name in enumerate(class_names):
        tp = np.sum((y_pred == i) & (y_true == i))
        fp = np.sum((y_pred == i) & (y_true != i))
        fn = np.sum((y_pred != i) & (y_true == i))
        support = np.sum(y_true == i)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        print(f"{name:<22} {precision:10.2f} {recall:10.2f} {f1:10.2f} {support:10d}")

# Early Stopping Rules
PATIENCE = 8                  # Consecutive epochs to tolerate an un-improving validation loss curve
MIN_DELTA = 1e-4              # Absolute minimal reduction required to qualify as an optimization step

# The encoder fills empty canvas with token END_PAD = 3 (asm_parser.TOKEN_MAP),
# never with PADDING = 0, so that "no instruction here" and "operand slot empty"
# stay distinct. Inserted rows must use the same value or the augmentation
# injects a token the model never sees at evaluation time.
END_PAD_PIXEL = 3 / 255.0

# ==============================================================================
# TRANSFORMS & STRUCTURAL MASK-AWARE AUGMENTATION
# ==============================================================================
class MaskAwareStructuralShift(object):
    """
    RESEARCH CORE UPGRADE: Updates both the image texture and the corresponding
    ViT 16x16 patch mask when simulating malware spatial layout expansion/shifting.
    """
    def __init__(self, p=0.4, max_shift_ratio=0.20, patch_size=16):
        self.p = p
        self.max_shift_ratio = max_shift_ratio
        self.patch_size = patch_size

    def __call__(self, sample):
        # Unpack the synchronized dictionary payload
        img_tensor, mask_tensor = sample['image'], sample['mask']
        if random.random() > self.p:
            return {'image': img_tensor, 'mask': mask_tensor}

        C, H, W = img_tensor.shape
        # Shift and insertion point in whole patch rows, so the image and the
        # 16x16 patch mask move by exactly the same amount (a pixel-row shift
        # with a rounded patch-row shift desynchronised them by up to a patch).
        shift_patches = max(1, int(np.ceil(H * random.uniform(0.05, self.max_shift_ratio) / self.patch_size)))
        insert_patch_row = random.randint(int(H * 0.1) // self.patch_size, int(H * 0.8) // self.patch_size)
        shift_amt = shift_patches * self.patch_size
        insert_row = insert_patch_row * self.patch_size

        # 1. Mutate the image matrix canvas layout
        top_half_img = img_tensor[:, :insert_row, :]
        bottom_half_img = img_tensor[:, insert_row:, :]
        padding_block = torch.full((C, shift_amt, W), END_PAD_PIXEL, dtype=img_tensor.dtype, device=img_tensor.device)
        shifted_img = torch.cat([top_half_img, padding_block, bottom_half_img], dim=1)[:, :H, :]

        # 2. The same shift on the mask track

        top_half_mask = mask_tensor[:insert_patch_row, :]
        bottom_half_mask = mask_tensor[insert_patch_row:, :]
        mask_padding = torch.zeros((shift_patches, mask_tensor.shape[1]), dtype=mask_tensor.dtype, device=mask_tensor.device)

        shifted_mask = torch.cat([top_half_mask, mask_padding, bottom_half_mask], dim=0)[:mask_tensor.shape[0], :]

        return {'image': shifted_img, 'mask': shifted_mask}


class MalwareMaskedDataset(Dataset):
    """Custom Data Engine loading companion matrix images and self-attention keys."""
    def __init__(self, base_folder, transform=None):
        self.base_folder = base_folder
        self.transform = transform

        self.all_samples = []
        self.class_names = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        for cls_name in self.class_names:
            cls_dir = os.path.join(base_folder, cls_name)
            for f in os.listdir(cls_dir):
                if f.lower().endswith('.png'):
                    self.all_samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        img_path, label = self.all_samples[idx]

        # 1. Load baseline PNG texture representation
        image = Image.open(img_path).convert('L')
        img_tensor = transforms.ToTensor()(image)

        # 2. Extract corresponding dynamic ViT mask track file
        # foo.png -> foo_vit_mask.npy
        mask_path = img_path[:-4] + "_vit_mask.npy"
        if os.path.exists(mask_path):
            mask_np = np.load(mask_path).astype(np.float32)
        else:
            # Fallback for unmasked samples: treat entire canvas as active code instructions
            mask_np = np.ones((16, 16), dtype=np.float32)

        mask_tensor = torch.tensor(mask_np)

        # 3. Synchronize structural augmentations if requested
        if self.transform:
            augmented = self.transform({'image': img_tensor, 'mask': mask_tensor})
            img_tensor, mask_tensor = augmented['image'], augmented['mask']

        return img_tensor, mask_tensor, label

# ==============================================================================
# SQUEEZE-AND-EXCITATION BLOCK FOR CHANNEL-WISE ATTENTION
# ==============================================================================
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        weight = self.fc(x).view(B, C, 1, 1)
        return x * weight


# ==============================================================================
# STOCHASTIC DEPTH LAYER FOR TRANSFORMER REGULARIZATION
# ==============================================================================
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize to 0 or 1
        return x.div(keep_prob) * random_tensor


# ==============================================================================
# REFINED STAGE 1: HYBRID CONVOLUTIONAL STEM WITH SE ATTENTION
# ==============================================================================
class ConvStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            SqueezeExcitation(64)
        )

    def forward(self, x):
        return self.stem(x)


# ==========================================
# RESIDUAL PATCH MERGING
# ==========================================
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.act = nn.GELU()

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.reduction(x)
        x = self.norm(x)
        x = self.act(x)
        return self.se(x + res)


# ==============================================================================
# MULTI-HEAD ATTENTION WITH RELATIVE POSITIONAL BIAS
# ==============================================================================
class RobustRelativeAttention(nn.Module):
    def __init__(self, dim, num_patches=256, heads=8, dropout=0.25):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.num_tokens = num_patches
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.num_tokens - 1), heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.arange(self.num_tokens)
        relative_coords = coords[:, None] - coords[None, :]
        relative_position_index = relative_coords + (self.num_tokens - 1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        rel_pos_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].reshape(
            self.num_tokens, self.num_tokens, self.heads
        )
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_pos_bias.unsqueeze(0)

        # Inject structural key masks directly into the attention block matrix
        if mask is not None:
            # PyTorch expectation: fill attention matrix cells to mask with -inf
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.to_out(out))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_patches=256, heads=8, mlp_dim=512, dropout=0.25, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RobustRelativeAttention(dim, num_patches=num_patches, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, key_padding_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=key_padding_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HierarchicalMalwareNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.25, drop_path_rate=0.0):
        super().__init__()
        self.stage1_cnn = ConvStem()
        self.stage2_downsample = PatchMerging(in_channels=64, out_channels=128)
        self.stage3_downsample = PatchMerging(in_channels=128, out_channels=256)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        self.pos_drop = nn.Dropout(p=dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 6)]

        self.vit_layers = nn.ModuleList([
            TransformerBlock(dim=256, num_patches=256, heads=8, mlp_dim=512, dropout=dropout, drop_path=dpr[i])
            for i in range(6)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, num_classes)
        )

    def forward(self, x, raw_patch_masks=None):
        x = self.stage1_cnn(x)
        x = self.stage2_downsample(x)
        x = self.stage3_downsample(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # Shape: (B, 256, 256) (No CLS token)

        # Build mask strictly for the 256 patches
        transformer_padding_mask = None
        if raw_patch_masks is not None:
            flat_masks = raw_patch_masks.flatten(start_dim=1) # (B, 256)
            transformer_padding_mask = (flat_masks == 0)

        x = self.pos_drop(x)

        for layer in self.vit_layers:
            x = layer(x, key_padding_mask=transformer_padding_mask)

        # TWEAK: Masked Global Average Pooling over authentic space only
        if raw_patch_masks is not None:
            # Multiply padding zone embeddings by 0 so they don't corrupt the mean calculation
            mask_multiplier = flat_masks.unsqueeze(-1) # (B, 256, 1)
            active_embeddings = x * mask_multiplier
            # Divide by the authentic count per sample instead of dividing by a flat 256
            authentic_counts = flat_masks.sum(dim=1, keepdim=True).clamp(min=1) # (B, 1)
            pooled_features = active_embeddings.sum(dim=1) / authentic_counts
        else:
            pooled_features = x.mean(dim=1)

        logits = self.mlp_head(pooled_features)
        return logits


# ==============================================================================
# MODIFIED EVALUATION UTILITY FUNCTION
# ==============================================================================
def evaluate_model(model, loader, criterion, device, class_names, final_eval=False):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    num_classes = len(class_names)
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, patch_masks, labels in loader:
            images = images.to(device)
            patch_masks = patch_masks.to(device)
            labels = labels.to(device)

            outputs = model(images, raw_patch_masks=patch_masks)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)

            _, predicted = torch.max(outputs, 1)

            for c in range(num_classes):
                class_mask = (labels == c)
                class_correct[c] += ((predicted == labels) & class_mask).sum().item()
                class_total[c] += class_mask.sum().item()

            if final_eval:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total_samples
    overall_accuracy = (class_correct.sum().item() / total_samples) * 100

    print("\n" + "-" * 50)
    print("PER-CLASS ACCURACY")
    print("-" * 50)
    for c in range(num_classes):
        if class_total[c] > 0:
            class_acc = (class_correct[c] / class_total[c]) * 100
            print(
                f"  {class_names[c]:<22}: {class_acc:6.2f}%  "
                f"[{int(class_correct[c])}/{int(class_total[c])}]"
            )
    print("-" * 50 + "\n")

    if final_eval:
        return avg_loss, overall_accuracy, all_labels, all_preds
    return avg_loss, overall_accuracy
