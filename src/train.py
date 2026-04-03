import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import (
    USE_FRAME_AWARE_GROUNDING,
    USE_CONTRASTIVE_ROI,
    USE_ENTITY_POOLING,
    CONTRASTIVE_TAU,
    validation,
    save_checkpoint_to_drive
)


def train_model(model, train_dataloader, val_dataloader, tokenizer, device,
                n_epochs=10, lr=0.001, log_file=None, checkpoint_name="checkpoint.pth"):
    """
    Sets up the optimization process:
    1. `criterion_images`: L1 Loss for image reconstruction.
    2. `criterion_ctx`: MSE Loss for context guidance (mean color).
    3. `criterion_text`: CrossEntropy Loss for text generation.
    4. `optimizer`: Adam optimizer for updating model weights.
    """

    criterion_images = nn.L1Loss()
    criterion_ctx = nn.MSELoss()
    criterion_text = nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- CoT-loss weights (added) ---
    LAMBDA_REID = 0.10  # pulls same-entity ROIs together (student idea)
    LAMBDA_GROUND_MSE = 0.10  # Option 2: frame-aware ROI↔text MSE grounding
    LAMBDA_CONTRAST = 0.10  # Option 1: contrastive ROI↔text grounding (InfoNCE)
    LAMBDA_ENTITY_POOL = 0.05  # Option 3: within-batch entity pooling loss

    # Log file for saving training results
    log_lines = []

    model.train()
    losses = []

    for epoch in range(n_epochs):

        running_loss = 0.0
        for (frames, descriptions, image_target, text_target,
             roi1, roi2, roi_valid, roi_frame, ent_id) in train_dataloader:

            # Send images and tokens to the GPU
            descriptions = descriptions.to(device)
            frames = frames.to(device)
            image_target = image_target.to(device)
            text_target = text_target.to(device)

            roi1 = roi1.to(device)
            roi2 = roi2.to(device)
            roi_valid = roi_valid.to(device)
            roi_frame = roi_frame.to(device)

            optimizer.zero_grad()

            # Predictions from our model (+ per-frame latents for CoT grounding)
            pred_image_content, pred_image_context, predicted_text_logits_k, _, _, z_v_seq, z_t_seq = model(
                frames, descriptions, text_target
            )

            # -------------------------
            # Base losses (unchanged)
            # -------------------------
            loss_im = criterion_images(pred_image_content, image_target)

            mu_global = frames.mean(dim=[0, 1])
            mu_global = mu_global.unsqueeze(0).expand_as(pred_image_context)
            loss_context = criterion_ctx(pred_image_context, mu_global)

            prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)
            target_labels = text_target.squeeze(1)[:, 1:]  # shift for teacher forcing
            target_flat = target_labels.reshape(-1)
            loss_text = criterion_text(prediction_flat, target_flat)

            # -------------------------
            # CoT-based grounding losses (added)
            # -------------------------
            loss_reid = torch.tensor(0.0, device=device)
            loss_ground_mse = torch.tensor(0.0, device=device)
            loss_contrast = torch.tensor(0.0, device=device)
            loss_entity_pool = torch.tensor(0.0, device=device)

            if roi_valid.any():
                mask = roi_valid.bool()
                if mask.sum() > 0:
                    z_r1 = model.image_encoder(roi1[mask])  # [M,D]
                    z_r2 = model.image_encoder(roi2[mask])  # [M,D]

                    # Simplest grounding: same entity across frames -> close in embedding
                    loss_reid = F.mse_loss(z_r1, z_r2)

                    # Option 2: frame-aware grounding MSE (ROI aligned to the description embedding of its frame)
                    if USE_FRAME_AWARE_GROUNDING:
                        f = roi_frame[mask].clamp(min=0, max=z_t_seq.size(1) - 1)  # [M]
                        z_t_match = z_t_seq[mask].gather(
                            1, f.view(-1, 1, 1).expand(-1, 1, z_t_seq.size(-1))
                        ).squeeze(1)  # [M,D]
                        loss_ground_mse = F.mse_loss(z_r1, z_t_match)

                    # Option 1: contrastive ROI↔text grounding (InfoNCE with batch negatives)
                    if USE_CONTRASTIVE_ROI and USE_FRAME_AWARE_GROUNDING:
                        # Normalize for cosine similarity
                        z_img = F.normalize(z_r1, dim=-1)
                        z_txt = F.normalize(z_t_match, dim=-1)
                        logits = (z_img @ z_txt.t()) / CONTRASTIVE_TAU  # [M,M]
                        labels = torch.arange(logits.size(0), device=device)
                        loss_contrast = F.cross_entropy(logits, labels)

                    # Option 3: entity-specific pooling/consistency across batch
                    if USE_ENTITY_POOLING:
                        # ent_id comes from the DataLoader as a list of strings
                        ent_list = [ent_id[i] for i, m in enumerate(mask.detach().cpu().tolist()) if m]
                        # group embeddings by entity id and pull to group mean (within-batch)
                        uniq = {}
                        for i_e, eid in enumerate(ent_list):
                            if not eid:
                                continue
                            uniq.setdefault(eid, []).append(i_e)

                        if len(uniq) > 0:
                            pool_losses = []
                            for eid, idxs in uniq.items():
                                if len(idxs) < 2:
                                    continue
                                group = z_r1[idxs]  # [G,D]
                                mean = group.mean(dim=0, keepdim=True)
                                pool_losses.append(F.mse_loss(group, mean.expand_as(group)))
                            if len(pool_losses) > 0:
                                loss_entity_pool = torch.stack(pool_losses).mean()

            # Total loss (base + optional improvements)
            loss = loss_im + loss_context + loss_text
            loss = loss + LAMBDA_REID * loss_reid
            loss = loss + LAMBDA_GROUND_MSE * loss_ground_mse
            loss = loss + LAMBDA_CONTRAST * loss_contrast
            loss = loss + LAMBDA_ENTITY_POOL * loss_entity_pool

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        losses.append(epoch_loss)

        print(
            f"Epoch [{epoch + 1}/{n_epochs}] Loss: {epoch_loss:.4f}  "
            f"(im={loss_im.item():.3f}, ctx={loss_context.item():.3f}, txt={loss_text.item():.3f}, "
            f"reid={float(loss_reid.detach()):.3f}, g_mse={float(loss_ground_mse.detach()):.3f}, "
            f"nce={float(loss_contrast.detach()):.3f}, entpool={float(loss_entity_pool.detach()):.3f})"
        )

        log_lines.append(f"Epoch [{epoch + 1}/{n_epochs}] Loss: {epoch_loss:.4f}  "
                         f"(im={loss_im.item():.3f}, ctx={loss_context.item():.3f}, txt={loss_text.item():.3f}, "
                         f"reid={float(loss_reid.detach()):.3f}, g_mse={float(loss_ground_mse.detach()):.3f}, "
                         f"nce={float(loss_contrast.detach()):.3f}, entpool={float(loss_entity_pool.detach()):.3f})")

        # Validation step
        validation(model, val_dataloader, tokenizer, device)
        save_checkpoint_to_drive(model, optimizer, epoch, epoch_loss, filename=checkpoint_name)
        model.train()  # Set back to train mode

    if log_file:
        with open(log_file, 'w') as f:
            f.write('\n'.join(log_lines))
        print(f"Training log saved to {log_file}")

    return losses
