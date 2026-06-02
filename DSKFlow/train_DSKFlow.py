import os
import gc
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy
import lpips
import torch
import wandb
from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict
from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
import vision_aided_loss
from model import make_1step_sched
from DSKFlow import DSKFlow, VAE_encode, VAE_decode, initialize_unet, initialize_vae,DegradationEncoder
from my_utils.training_utils import UnpairedDataset, build_transform, parse_args_unpaired_training
from my_utils.dino_struct import DinoStructureLoss


def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with=args.report_to)
    set_seed(args.seed)

    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint)
        resume_from_checkpoint = args.resume_from_checkpoint
        print(f"resume_from_checkpoint: {resume_from_checkpoint}")

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=args.revision, use_fast=False,)

    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()

    unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = initialize_unet(args.lora_rank_unet, return_lora_module_names=True)
    vae_a2b, vae_lora_target_modules = initialize_vae(args.lora_rank_vae, return_lora_module_names=True)

    # --- 训练脚本中的初始化修改 ---
    deg_encoder = DegradationEncoder(num_tokens=4, embed_dim=1024).cuda()

    queue_size = 64
    # deg_queue 存退化特征 (1024维)
    deg_encoder.register_buffer("deg_queue", torch.randn(queue_size, 1024).cuda())
    # v_queue 存速度场特征 (VAE Latent 是 4 通道)
    # 修改 v_queue 的维度，从 4 提升到 128
    v_feat_dim = 64
    deg_encoder.register_buffer("v_queue", torch.randn(queue_size, v_feat_dim).cuda())
    deg_encoder.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long).cuda())

    # 初始归一化
    deg_encoder.deg_queue = torch.nn.functional.normalize(deg_encoder.deg_queue, dim=1)
    deg_encoder.v_queue = torch.nn.functional.normalize(deg_encoder.v_queue, dim=1)

    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    if args.gan_disc_type == "vagan_clip":
        net_disc_a = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_a.cv_ensemble.requires_grad_(False)  # Freeze feature extractor
        net_disc_b = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_b.cv_ensemble.requires_grad_(False)  # Freeze feature extractor

    crit_cycle, crit_idt = torch.nn.L1Loss(), torch.nn.L1Loss()




    unet.conv_in.requires_grad_(True)
    vae_b2a = copy.deepcopy(vae_a2b)

    params_gen = DSKFlow.get_traininable_params(unet, vae_a2b, vae_b2a)
    params_gen += list(deg_encoder.parameters())  # 加入退化编码器参数

    vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)

    if args.resume_from_checkpoint:
        resume_from_checkpoint = args.resume_from_checkpoint
        print(f"resume_from_checkpoint: {resume_from_checkpoint}")

        if os.path.isfile(resume_from_checkpoint):
            print(f"resume_from_checkpoint: {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint, map_location='cpu')

            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(unet, checkpoint['sd_encoder'], adapter_name="default_encoder")
            set_peft_model_state_dict(unet, checkpoint['sd_decoder'], adapter_name="default_decoder")
            set_peft_model_state_dict(unet, checkpoint['sd_other'], adapter_name="default_others")

            vae_enc.load_state_dict(checkpoint['sd_vae_enc'])
            vae_dec.load_state_dict(checkpoint['sd_vae_dec'])
            deg_encoder.load_state_dict(checkpoint['deg_encoder'])

            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
                print(f"global_step: {global_step}")
            if 'epoch' in checkpoint:
                first_epoch = checkpoint['epoch'] + 1
                print(f"epoch: {first_epoch}")

            print(f"Successful")
        else:
            print(f"warning: {resume_from_checkpoint}")

    optimizer_gen = torch.optim.AdamW(params_gen, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)

    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_disc = torch.optim.AdamW(params_disc, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)

    dataset_train = UnpairedDataset(dataset_folder=args.dataset_folder, image_prep=args.train_img_prep, split="train", tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    T_val = build_transform(args.val_img_prep)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt
    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(args.dataset_folder, "test_A", ext)))
    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_tgt_test.extend(glob(os.path.join(args.dataset_folder, "test_B", ext)))
    l_images_src_test, l_images_tgt_test = sorted(l_images_src_test), sorted(l_images_tgt_test)

    # make the reference FID statistics
    if accelerator.is_main_process:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
        """
        FID reference statistics for A -> B translation
        """
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_a2b")
        os.makedirs(output_dir_ref, exist_ok=True)
        # transform all images according to the validation transform and save them
        for _path in tqdm(l_images_tgt_test):
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img = T_val(Image.open(_path).convert("RGB"))
                _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
        a2b_ref_mu, a2b_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
        """
        FID reference statistics for B -> A translation
        """
        # transform all images according to the validation transform and save them
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_b2a")
        os.makedirs(output_dir_ref, exist_ok=True)
        for _path in tqdm(l_images_src_test):
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img = T_val(Image.open(_path).convert("RGB"))
                _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
        b2a_ref_mu, b2a_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)

    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)

    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.cuda()
    net_lpips.requires_grad_(False)

    fixed_a2b_tokens = tokenizer(fixed_caption_tgt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.cuda().unsqueeze(0))[0].detach()
    fixed_b2a_tokens = tokenizer(fixed_caption_src, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.cuda().unsqueeze(0))[0].detach()
    del text_encoder, tokenizer  # free up some memory

    unet, vae_enc, vae_dec, deg_encoder, net_disc_a, net_disc_b = accelerator.prepare(
        unet, vae_enc, vae_dec, deg_encoder, net_disc_a, net_disc_b
    )
    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc)


    first_epoch = 0
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)
    # turn off eff. attn for the disc
    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_b.named_modules():
        if "attn" in name:
            module.fused_attn = False

    for epoch in range(first_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            l_acc = [unet, net_disc_a, net_disc_b, vae_enc, vae_dec]
            with accelerator.accumulate(*l_acc):
                img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)



                bsz = img_a.shape[0]
                fixed_a2b_emb = fixed_a2b_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                fixed_b2a_emb = fixed_b2a_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)

                # ==========================================
                # Generator Phase: Unified Forward & Backward
                # ==========================================
                optimizer_gen.zero_grad()

                # 1. Extract Anchor Condition (Only once, from img_a)
                deg_tokens_anchor, deg_mask_a = deg_encoder(img_a)

                # 2. Physics Objectives (y is not None -> returns tuple of losses)
                loss_fm_a2b, loss_ke_a2b, loss_div_a2b, loss_mi_a2b, v_snap_a, d_snap_a = DSKFlow.forward_with_networks(
                    x=img_a, y=img_b, direction="a2b", vae_enc=vae_enc, unet=unet, vae_dec=vae_dec,
                    text_emb=fixed_a2b_emb, deg_encoder=deg_encoder, deg_tokens=deg_tokens_anchor, deg_mask=deg_mask_a,
                )
                loss_fm_b2a, loss_ke_b2a, loss_div_b2a, loss_mi_b2a, v_snap_b, d_snap_b = DSKFlow.forward_with_networks(
                    x=img_b, y=img_a, direction="b2a", vae_enc=vae_enc, unet=unet, vae_dec=vae_dec,
                    text_emb=fixed_b2a_emb, deg_encoder=deg_encoder, deg_tokens=deg_tokens_anchor, deg_mask=deg_mask_a,
                )

                loss_fm = (loss_fm_a2b + loss_fm_b2a) * args.lambda_fm
                loss_ke = (loss_ke_a2b + loss_ke_b2a) * args.lambda_ke
                loss_div = (loss_div_a2b + loss_div_b2a) * args.lambda_div
                loss_mi = (loss_mi_a2b + loss_mi_b2a) * args.lambda_mi
                loss_physics = loss_fm + loss_ke + loss_div + loss_mi

                # 3. Cycle & Generative Objectives (y is None -> returns image)
                # A -> fake B -> rec A
                fake_b = DSKFlow.forward_with_networks(
                    x=img_a, y=None, direction="a2b", vae_enc=vae_enc, unet=unet, vae_dec=vae_dec,
                    text_emb=fixed_a2b_emb, deg_encoder=deg_encoder, deg_tokens=deg_tokens_anchor,deg_mask=deg_mask_a,
                )
                rec_a = DSKFlow.forward_with_networks(
                    x=fake_b, y=None, direction="b2a", vae_enc=vae_enc, unet=unet, vae_dec=vae_dec,
                    text_emb=fixed_b2a_emb, deg_encoder=deg_encoder, deg_tokens=deg_tokens_anchor,deg_mask=deg_mask_a,
                )
                loss_cycle_a = crit_cycle(rec_a, img_a) * args.lambda_cycle + net_lpips(rec_a,
                                                                                        img_a).mean() * args.lambda_cycle_lpips

                # B -> fake A -> rec B
                fake_a = DSKFlow.forward_with_networks(
                    x=img_b, y=None, direction="b2a", vae_enc=vae_enc, unet=unet, vae_dec=vae_dec,
                    text_emb=fixed_b2a_emb, deg_encoder=deg_encoder, deg_tokens=deg_tokens_anchor,deg_mask=deg_mask_a,
                )
                rec_b = DSKFlow.forward_with_networks(
                    x=fake_a, y=None, direction="a2b", vae_enc=vae_enc, unet=unet, vae_dec=vae_dec,
                    text_emb=fixed_a2b_emb, deg_encoder=deg_encoder, deg_tokens=deg_tokens_anchor,deg_mask=deg_mask_a,
                )
                loss_cycle_b = crit_cycle(rec_b, img_b) * args.lambda_cycle + net_lpips(rec_b,
                                                                                        img_b).mean() * args.lambda_cycle_lpips

                # 4. GAN Objectives
                loss_gan_a = net_disc_a(fake_b, for_G=True).mean() * args.lambda_gan
                loss_gan_b = net_disc_b(fake_a, for_G=True).mean() * args.lambda_gan

                # 5. Identity Objective

                idt_a = DSKFlow.forward_with_networks(
                    x=img_b, y=None, direction="a2b", vae_enc=vae_enc, unet=unet, vae_dec=vae_dec,
                    text_emb=fixed_a2b_emb, deg_encoder=deg_encoder, deg_tokens=deg_tokens_anchor,deg_mask=deg_mask_a,
                )
                loss_idt_a = crit_idt(idt_a, img_a) * args.lambda_idt

                idt_b = DSKFlow.forward_with_networks(
                    x=img_a, y=None, direction="b2a", vae_enc=vae_enc, unet=unet, vae_dec=vae_dec,
                    text_emb=fixed_a2b_emb, deg_encoder=deg_encoder, deg_tokens=deg_tokens_anchor, deg_mask=deg_mask_a,
                )
                loss_idt_b = crit_idt(idt_b, img_b) * args.lambda_idt

                # ==========================================
                # One Unified Generator Backward
                # ==========================================
                total_gen_loss =  loss_gan_a + loss_gan_b + loss_cycle_b + loss_cycle_a +loss_physics + loss_idt_a + loss_idt_b

                accelerator.backward(total_gen_loss, retain_graph=False)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)

                optimizer_gen.step()
                lr_scheduler_gen.step()

                # ==========================================
                # Discriminator Phase
                # ==========================================

                """
                Discriminator for task a->b and b->a (fake inputs)
                """
                loss_D_A_fake = net_disc_a(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                loss_D_B_fake = net_disc_b(fake_a.detach(), for_real=False).mean() * args.lambda_gan
                loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5
                accelerator.backward(loss_D_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                """
                Discriminator for task a->b and b->a (real inputs)
                """
                loss_D_A_real = net_disc_a(img_b, for_real=True).mean() * args.lambda_gan
                loss_D_B_real = net_disc_b(img_a, for_real=True).mean() * args.lambda_gan
                loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5
                accelerator.backward(loss_D_real, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                # ==========================================
                # Logging
                # ==========================================
                logs = {
                    "loss_fm": loss_fm.detach().item(),
                    "loss_ke": loss_ke.detach().item(),
                    "loss_div": loss_div.detach().item(),
                    "loss_mi": loss_mi.detach().item(),
                    "cycle": loss_cycle_a.detach().item(),
                    "gan": loss_gan_a.detach().item(),
                    "disc": loss_D_A_fake.detach().item() + loss_D_A_real.detach().item(),

                }

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    eval_unet = accelerator.unwrap_model(unet)
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)
                    eval_deg_encoder = accelerator.unwrap_model(deg_encoder)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        sd = {}
                        sd["l_target_modules_encoder"] = l_modules_unet_encoder
                        sd["l_target_modules_decoder"] = l_modules_unet_decoder
                        sd["l_modules_others"] = l_modules_unet_others
                        sd["rank_unet"] = args.lora_rank_unet
                        sd["sd_encoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_encoder")
                        sd["sd_decoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_decoder")
                        sd["sd_other"] = get_peft_model_state_dict(eval_unet, adapter_name="default_others")
                        sd["rank_vae"] = args.lora_rank_vae
                        sd["vae_lora_target_modules"] = vae_lora_target_modules
                        sd["sd_vae_enc"] = eval_vae_enc.state_dict()
                        sd["sd_vae_dec"] = eval_vae_dec.state_dict()
                        sd["deg_encoder"] = eval_deg_encoder.state_dict() # 保存自适应编码器
                        torch.save(sd, outf)
                        gc.collect()
                        torch.cuda.empty_cache()

                    # compute val FID and DINO-Struct scores
                    if global_step % args.validation_steps == 1:

                        net_dino = DinoStructureLoss()
                        """
                        Evaluate "A->B"
                        """
                        fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples_a2b")
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_a2b = []
                        # get val input images from domain a
                        for idx, input_img_path in enumerate(tqdm(l_images_src_test)):
                            if idx > args.validation_num_images and args.validation_num_images > 0:
                                break
                            outf = os.path.join(fid_output_dir, f"{idx}.png")
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                img_a = transforms.ToTensor()(input_img)
                                img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda()
                                deg_tokens_val, deg_mask_val = deg_encoder(img_a)
                                
                                eval_fake_b = DSKFlow.forward_with_networks(img_a,None, "a2b", eval_vae_enc, eval_unet,
                                    eval_vae_dec,  fixed_a2b_emb[0:1],deg_encoder,deg_tokens_val,deg_mask_val)
                                eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                                eval_fake_b_pil.save(outf)
                                a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                b = net_dino.preprocess(eval_fake_b_pil).unsqueeze(0).cuda()
                                dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                l_dino_scores_a2b.append(dino_ssim)
                        dino_score_a2b = np.mean(l_dino_scores_a2b)
                        gen_features = get_folder_features(fid_output_dir, model=feat_model, num_workers=0, num=None,
                            shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                            mode="clean", custom_fn_resize=None, description="", verbose=True,
                            custom_image_tranform=None)
                        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                        score_fid_a2b = frechet_distance(a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
                        print(f"step={global_step}, fid(a2b)={score_fid_a2b:.2f}, dino(a2b)={dino_score_a2b:.3f}")

                        logs["val/fid_a2b"] = score_fid_a2b
                        logs["val/dino_struct_a2b"] = dino_score_a2b
                        del net_dino  # free up memory

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    args = parse_args_unpaired_training()
    main(args)
