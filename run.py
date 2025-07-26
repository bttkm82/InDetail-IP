import torch
from tqdm import tqdm
import click
import os

from configs import walk_configs, save_config
from src.utils import get_logger, logging_info, seed_everything, load_img, init_ip_model

@click.command()
@click.option("--default_setting", type=str, default="./configs/default_settings.yaml", help="default setting file")
@click.option("--c", type=str, required=True, help="Path to the inference configuration file.")
def main(default_setting, c): # d): #r, t, ns, f, s, img_scale, ea, do_stop):
    dft_args = walk_configs(default_setting)
    args = walk_configs(c)
    seed_everything(args.seed)
    img_sz = dft_args.models[args.model].img_sz

    args.exp_folder = f"{args.root_dir}/{args.exp_folder}"
    os.makedirs(f"{args.exp_folder}/image_out", exist_ok=True)
    os.makedirs(f"{args.exp_folder}/recon", exist_ok=True)
    os.makedirs(f"{args.root_dir}/0_input_conditions/in_image_{img_sz}", exist_ok=True)

    if hasattr(args, 'cond'):
        img_name = f"{os.path.splitext(os.path.basename(args.img_path))[0]}"
        condition = load_img(args.cond_path, img_sz=img_sz)
    else:
        img_name = os.path.splitext(os.path.basename(args.img_path))[0]

    image_gt = load_img(args.img_path, img_sz=img_sz)
    in_img_path = f"{args.root_dir}/0_input_conditions/in_image_{img_sz}/{img_name}.png"
    if not os.path.exists(in_img_path):
        image_gt.save(in_img_path)

    get_logger(f"{args.exp_folder}/logging.log", force_add_handler=True)
    save_config(f"{args.exp_folder}/config.yaml", args)

    ip_model = init_ip_model(args, dft_args)
    logging_info(f"{args}")

    prefix = ", best quality, extremely detailed."

    for idx_cond in tqdm(range(len(args.inf_text))):
        prompt = args.inf_text[idx_cond]

        neg_prompt = 'bad anatomy, bad hands, cropped, worst quality'
        inv_prompt = args.inv_text

        kwargs = {
            "pil_image": image_gt,
            "num_samples": args.num_samples,
            "prompt": f"{prompt}{prefix}" if args.use_prefix else prompt,
            "negative_prompt": neg_prompt,
            "num_inference_steps": args.ddim_step,
            "guidance_scale": args.cfg,
            "inv_prompt": f"{inv_prompt}{prefix}" if args.use_prefix else inv_prompt,
        }
        if hasattr(args, 'cond'):
            kwargs.update({'image': condition})

        with torch.no_grad():
            torch.cuda.empty_cache()
            images, rec_image = ip_model.generate(**kwargs)
            if idx_cond == 0:
                rec_image.save(f"{args.exp_folder}/recon/{img_name}.png")

        ## save results
        for img_idx, im in enumerate(images):
            if hasattr(args, 'cond'):
                im.save(f"{args.exp_folder}/image_out/{img_name}_{args.cond}_{idx_cond:0>4}_{img_idx:0>3}.png")
            else:
                im.save(f"{args.exp_folder}/image_out/{img_name}_{idx_cond:0>4}_{img_idx:0>3}.png")

    print(f"Save results: {args.exp_folder}")

if __name__ == "__main__":
    main()
