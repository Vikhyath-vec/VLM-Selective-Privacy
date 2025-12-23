import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from einops import rearrange
import scipy.io
import sys
from pathlib import Path
sys.path.append('../Transformer-Explainability-main')
from tqdm import tqdm
from modules.layers_ours import *
from baselines.ViT.helpers import load_pretrained
from baselines.ViT.weight_init import trunc_normal_
from baselines.ViT.layer_helpers import to_2tuple
from baselines.ViT.ViT_LRP import Mlp, PatchEmbed
from transformers import LlavaForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.k_proj = Linear(dim, dim, bias=qkv_bias)
        self.q_proj = Linear(dim, dim, bias=qkv_bias)
        self.v_proj = Linear(dim, dim, bias=qkv_bias)

        self.matmul_qk = einsum('bhid,bhjd->bhij')
        self.matmul_av = einsum('bhij,bhjd->bhid')

        self.attn_drop = Dropout(attn_drop)
        self.out_proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None
    
    def get_attn(self): return self.attn
    def set_attn(self, x): self.attn = x
    def get_attn_cam(self): return self.attn_cam
    def set_attn_cam(self, x): self.attn_cam = x
    def get_v(self): return self.v
    def set_v(self, x): self.v = x
    def get_v_cam(self): return self.v_cam
    def set_v_cam(self, x): self.v_cam = x
    def get_attn_gradients(self): return self.attn_gradients
    def set_attn_gradients(self, x): self.attn_gradients = x
    
    def forward(self, x):
        b, n, c = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        self.set_v(v)
        dots = self.matmul_qk([q, k]) * self.scale
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)
        self.set_attn(attn)
        if attn.requires_grad:
            attn.retain_grad()
            # attn.register_hook(self.set_attn_gradients)
            attn.register_hook(lambda grad: setattr(self, "attn_gradients", grad))
        
        out = self.matmul_av([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out
    
    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.out_proj.relprop(cam, **kwargs)

        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)
        cam_attn, cam_v = self.matmul_av.relprop(cam, **kwargs)
        cam_attn /= 2
        cam_v /= 2
        self.set_attn_cam(cam_attn)
        self.set_v_cam(cam_v)

        cam_attn = self.attn_drop.relprop(cam_attn, **kwargs)
        cam_attn = self.softmax.relprop(cam_attn, **kwargs)
        cam_q, cam_k = self.matmul_qk.relprop(cam_attn, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_q = rearrange(cam_q, 'b h n d -> b n (h d)')
        cam_k = rearrange(cam_k, 'b h n d -> b n (h d)')
        cam_v = rearrange(cam_v, 'b h n d -> b n (h d)')
        cam_q = self.q_proj.relprop(cam_q, **kwargs)
        cam_k = self.k_proj.relprop(cam_k, **kwargs)
        cam_v = self.v_proj.relprop(cam_v, **kwargs)

        cam = cam_q + cam_k + cam_v
        return cam

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-5)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = LayerNorm(dim, eps=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()
    
    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x
    
    def relprop(self, cam, **kwargs):
        cam1, cam2 = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)
        cam1, cam2 = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam

class VisionModel(nn.Module):
    def __init__(self, img_size=336, patch_size=14, in_channels=3, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Embedding(num_patches + 1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        self.pre_layernorm = LayerNorm(embed_dim, eps=1e-5)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.post_layernorm = LayerNorm(embed_dim, eps=1e-5)

        self.add = Add()
        self.inp_grad = None
        self.second_last_hidden = None
        self.last_hidden = None

    def get_inp_grad(self): return self.inp_grad
    def set_inp_grad(self, x): self.inp_grad = x

    @property
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}
    
    def forward(self, pixel_values, output_hidden_states=False, return_dict=True, output_attentions=False, interpolate_pos_encoding=False):
        x = pixel_values
        hidden_states = []
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_ids = torch.arange(x.shape[1], device=x.device)
        x = self.add([x, self.pos_embed(pos_ids)])
        x = self.pre_layernorm(x)
        hidden_states.append(x)
        for block in self.blocks:
            x = block(x)
            if output_hidden_states:
                hidden_states.append(x)
        self.second_last_hidden = x
        if x.requires_grad:
            x.register_hook(self.set_inp_grad)
        x = self.post_layernorm(x)
        hidden_states.append(x)
        self.last_hidden = x
        pooled = x[:, 0]
        if return_dict:
            return BaseModelOutputWithPooling(
                last_hidden_state=self.last_hidden,
                pooler_output=pooled,
                hidden_states=tuple(hidden_states),
            )
        return self.last_hidden, pooled, hidden_states
    
    def relprop(self, cam, method="transformer_attribution", start_layer=0, **kwargs):
        cam = self.post_layernorm.relprop(cam, **kwargs)
        for block in reversed(self.blocks):
            cam = block.relprop(cam, **kwargs)
        cam = self.pre_layernorm.relprop(cam, **kwargs)
        
        if method == "full":
            cam, _ = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            cam = cam.sum(dim=1)
            return cam
        
        elif method == "transformer_attribution":
            cams = []
            for block in self.blocks:
                grad = block.attn.get_attn_gradients()
                cam = block.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam

class VisionModelWithHead(nn.Module):
    def __init__(self, vm, num_classes=1000):
        super().__init__()
        self.vm = vm
        self.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        outputs = self.vm(x)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits

def remap_hf_to_custom(hf_sd):
    new_sd = {}
    for k, v in hf_sd.items():
        if k.startswith("embeddings.patch_embedding"):
            new_k = k.replace("embeddings.patch_embedding", "patch_embed.proj")
        elif k.startswith("embeddings.position_embedding"):
            new_k = k.replace("embeddings.position_embedding", "pos_embed")
        elif k.startswith("pre_layrnorm"):
            new_k = k.replace("pre_layrnorm", "pre_layernorm")
        elif k.startswith("post_layernorm"):
            new_k = k.replace("post_layernorm", "post_layernorm")
        elif k.startswith("encoder.layers"):
            new_k = (
                k.replace("encoder.layers", "blocks")
                .replace("self_attn", "attn")
                .replace("layer_norm1", "norm1")
                .replace("layer_norm2", "norm2")
                .replace("mlp.fc1", "mlp.fc1")
                .replace("mlp.fc2", "mlp.fc2")
            )
        else:
            continue
        new_sd[new_k] = v
    return new_sd

def get_llava_model_with_custom_vision(model_id="llava-hf/llava-1.5-7b-hf", device_map="cpu"):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map=device_map
    )

    vision_model = model.vision_tower.vision_model
    hf_sd = vision_model.state_dict()
    
    vm = VisionModel()
    
    mapped_sd = remap_hf_to_custom(hf_sd)
    missing, unexpected = vm.load_state_dict(mapped_sd, strict=False)
    
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
        
    return model, vm

class ImagenetValDataset(Dataset):
    def __init__(self, val_dir, gt_txt, transform=None):
        self.val_dir = val_dir
        self.gt_txt = gt_txt
        self.transform = transform
        with open(gt_txt, "r") as f:
            self.labels = [int(x.strip()) - 1 for x in f.readlines()]
        self.images = sorted([
            fname for fname in os.listdir(val_dir)
            if fname.endswith(".JPEG")
        ])
        assert len(self.images) == len(self.labels), "Number of images and labels must match"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.val_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def train_one_epoch(model, dataloader, idx2label, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, folder_labels in tqdm(dataloader):
        images = images.to(device)
        folder_labels = folder_labels.to(device)
        labels = idx2label[folder_labels].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples

def train_and_validate(model, train_dataloader, val_dataloader, idx2label_train, optimizer, criterion, device, num_epochs=10, save_path="model.pth"):
    idx2label_train = idx2label_train.to(device)
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_dataloader, idx2label_train, optimizer, criterion, device)
        val_loss, val_acc = validate_one_epoch(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch}/{num_epochs}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path
        )
        print(f"Saved model at epoch {epoch} to {save_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vm = get_llava_model_with_custom_vision(device_map=device)
    vm.requires_grad_(False)
    for param in vm.parameters():
        param.requires_grad = False
    vm_with_head = VisionModelWithHead(vm, num_classes=1000).to(device)
    optim = torch.optim.Adam(vm_with_head.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    IMAGENET_PATH = Path("/ocean/projects/phy250048p/kothamas/data")

    train_transforms = transforms.Compose([
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    train_dataset_path = IMAGENET_PATH / "ILSVRC2012_img_train"
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    meta = scipy.io.loadmat(IMAGENET_PATH / "ILSVRC2012_devkit_t12/data/meta.mat")
    synsets = meta['synsets']
    wnid_to_imagenet = {}
    for s in synsets[:1000]:
        imagenet_id = int(s[0][0][0][0])
        wnid = str(s[0][1][0])
        wnid_to_imagenet[wnid] = imagenet_id - 1
    folder_to_idx = train_dataset.class_to_idx
    num_classes = len(folder_to_idx)
    idx_to_imagenet = torch.empty(num_classes, dtype=torch.long)
    for wnid, folder_idx in folder_to_idx.items():
        idx_to_imagenet[folder_idx] = wnid_to_imagenet[wnid]

    val_transforms = transforms.Compose([
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    val_dataset = ImagenetValDataset(
        IMAGENET_PATH / "ILSVRC2012_img_val",
        IMAGENET_PATH / "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
        val_transforms
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    train_and_validate(
        vm_with_head,
        train_loader,
        val_loader,
        idx_to_imagenet,
        optim,
        criterion,
        device,
        num_epochs=10,
        save_path="vm_with_head.pth"
    )

    # model, vm = get_llava_model_with_custom_vision()
    # x = torch.randn(1, 3, 336, 336)
    # out = vm(x)
    # R = torch.randn_like(out)
    # vm.zero_grad()
    # out.backward(gradient=R, retain_graph=True)
    # cam = vm.relprop(R, alpha=1)
    # print(cam.shape)