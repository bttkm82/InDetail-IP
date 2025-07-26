import torch
import numpy as np
import importlib
import torch.fft

from PIL import Image
import random
import logging
import sys
logging.getLogger("PIL").setLevel(logging.WARNING)  # avoid PIL report


def PILtoTensor(data: Image.Image) -> torch.Tensor:
    return torch.tensor(np.array(data)).permute(2, 0, 1).unsqueeze(0).float()

def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_logger(path_log="logging.log", force_add_handler=False):
    """
    Set up the logger. Note that the setting will also impact the default logging logger, which means that simply
    using logging.info() will output the logs to both stdout and the filename_log.
    :param path_log: the filename of the log
    :param force_add_handler: if True, will clear logging.root.handlers
    :type path_log: str
    """
    ret_logger = logging.getLogger()
    ret_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s: %(message)s", datefmt="%Y-%m-%d-%H:%M:%S"
    )

    if force_add_handler:
        ret_logger.handlers = []

    if not ret_logger.handlers:
        path_log = "%s.log" % path_log if not path_log.endswith(".log") else path_log
        fh = logging.FileHandler(path_log)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        ret_logger.addHandler(ch)
        ret_logger.addHandler(fh)

    return ret_logger


def logging_info(*args):
    if logging.root.level > logging.getLevelName("INFO"):
        logging.warning("Logging level higher than INFO!")
        print(*args)
    else:
        logging.info(*args)

def import_module(module_path):
    module_name = module_path.split(".")
    pkg_name, module_name = module_name[:-1], module_name[-1]
    pkg_name = ".".join(pkg_name)
    im_module = importlib.import_module(pkg_name)
    im_module = getattr(im_module, module_name)
    return im_module

def load_img(image_path, left=0, right=0, top=0, bottom=0, img_sz=512):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert("RGB"))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = Image.fromarray(image).resize((img_sz, img_sz))
    return image



def create_frequency_mask(shape, radius, low_pass=True):
    """
    Creates a circular mask for low-pass or high-pass filtering.
    - shape: tuple (H, W), the shape of the frequency domain image.
    - radius: int, the radius for the low-pass or high-pass filter.
    - low_pass: bool, if True creates a low-pass mask, otherwise high-pass.
    """
    H, W = shape
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_y, center_x = H // 2, W // 2
    dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    if low_pass:
        mask = dist <= radius
    else:
        mask = dist > radius

    return mask


def apply_filter_to_channel(channel, radius, low_pass=True):
    # Step 1: Apply 2D FFT
    freq_domain = torch.fft.fft2(channel)
    freq_shifted = torch.fft.fftshift(freq_domain)

    # Step 2: Create the mask
    mask = create_frequency_mask(freq_shifted.shape, radius, low_pass=low_pass)

    # Step 3: Apply the mask
    filtered_freq = freq_shifted * mask

    # Step 4: Inverse FFT to get the filtered image
    filtered_freq_shifted = torch.fft.ifftshift(filtered_freq)
    filtered_image = torch.fft.ifft2(filtered_freq_shifted).real

    return filtered_image


def get_freq_image(image: torch.FloatTensor, radius: int, low_pass=True):
    """
    Process an RGB image, applying low-pass or high-pass filter to each channel.
    - image: 3D tensor of shape (3, H, W)
    - radius: int, filter radius
    - low_pass: bool, True for low-pass, False for high-pass
    """
    return torch.stack([apply_filter_to_channel(img, radius, low_pass=low_pass) for img in image])

from diffusers import DDIMScheduler, ControlNetModel

def init_ip_model(args, dft_args):
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False, steps_offset=1)
    device = "cuda"
    dtype = torch.float16 if args.is_half else torch.float32
    pipe_module = import_module(args.pipe)

    if hasattr(args, 'cond'):
        controlnet_model_path = dft_args.controlnet_model_path[args.model][args.cond]
        controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=dtype,
                                                     cache_dir=args.cache_dir)

        pipe = pipe_module.from_pretrained(
            dft_args.models[args.model].pretrained_model_path,
            controlnet=controlnet,
            torch_dtype=dtype,
            scheduler=scheduler,
            cache_dir=args.cache_dir,
            safety_checker=None,
        ).to(device)
    else:
        pipe = pipe_module.from_pretrained(
            dft_args.models[args.model].pretrained_model_path, torch_dtype=dtype,
            scheduler=scheduler,
            cache_dir=args.cache_dir,
            safety_checker=None,
        ).to(device)

    pipe.set_progress_bar_config(leave=False)
    pipe.set_progress_bar_config(disable=True)
    ip_model_module = import_module(args.ip_model)
    ip_model = ip_model_module(pipe, args, device, dtype=dtype)
    return ip_model