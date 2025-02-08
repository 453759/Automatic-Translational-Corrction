from utils import macher
from project_config.match_config import get_args
import torchvision.transforms as transforms
from skimage.io import imread
from skimage.color import gray2rgb
from lib.normalization import imreadth, resize, normalize
import torch
import numpy as np

def imreadth_with_rgb_conversion(image_path):
    # Read the image
    image = imread(image_path).astype(np.float32)

    # If the image is grayscale, convert it to RGB format
    if len(image.shape) == 2:  # If the image is grayscale (H x W)
        image = gray2rgb(image)  # Convert to RGB format (H x W x 3)

    # Convert to PyTorch tensor format (C x H x W)
    return torch.Tensor(image).permute(2, 0, 1)  # Convert the channel order

use_cuda = torch.cuda.is_available()

im_fe_ratio = 16
half_precision = True

torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

Transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

args = get_args()
torch.cuda.set_device(args.device)
feature_extractor_device = args.device

matcher = macher.ImgMatcher(use_cuda=use_cuda, half_precision=half_precision, checkpoint=args.checkpoint, postprocess_device=feature_extractor_device, im_fe_ratio=args.im_fe_ratio)
scale_factor = 0.0625
running_time = 0
counter = 0

def get_match_points(query_im_pth, ref_im_pth):
    query_im = imreadth_with_rgb_conversion(query_im_pth)
    hA, wA = query_im.shape[-2:]
    query_im = resize(normalize(query_im), args.image_size, scale_factor)
    hA_, wA_ = query_im.shape[-2:]

    ref_im = imreadth_with_rgb_conversion(ref_im_pth)
    hB, wB = ref_im.shape[-2:]
    ref_im = resize(normalize(ref_im), args.image_size, scale_factor)
    hB_, wB_ = ref_im.shape[-2:]

    # Create batch
    batch = {}
    batch['source_image'] = query_im.cuda()
    batch['target_image'] = ref_im.cuda()

    matches, score, _ = matcher(batch, num_pts=args.Npts, central_align=True, iter_step=args.iter_step)
    matches = matches.cpu().numpy()
    score = score.detach().view(-1).cpu().numpy()

    query = matches[:, :2] * (hA / hA_)  # Keypoint coordinates in image A
    ref = matches[:, 2:] * (hB / hB_)  # Keypoint coordinates in image B
    return query, ref

