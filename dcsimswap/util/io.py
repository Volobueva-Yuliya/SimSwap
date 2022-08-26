import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from bisenet import BiSeNetModel
from skvideo.io import FFmpegWriter
import os
import uuid
import subprocess
from skimage import io
from tqdm.auto import tqdm
from pathlib import Path
from dcsimswap.insightface_func.face_detect_crop_multi import Face_detect_crop
from dcsimswap.models.models import create_model
from dcsimswap.options.test_options import TestOptions


DEVICE = torch.device("cuda:0")

def video_swap(id_vetor, id_vetor_flip, debug, debug_max_frames, face_detection_model, video, debug_writer, writer, swap_model, use_image_flip, parsing_model, crop_size=512):
    """Read video and infer SimSwap"""

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    with tqdm(total=frame_count) as bar:
        while True:
            if debug:
                if frame_index == debug_max_frames:
                    break
            success, frame = video.read()
            if not success:
                break

            detect_results = face_detection_model.get(frame, crop_size)
            if detect_results is not None:  # face detected
                align_faces = detect_results[0]
                swap_result_list = infer_simswap_and_psfrgan(align_faces, id_vetor, id_vetor_flip, swap_model, use_image_flip)

                oriimg, debug_img, final_img = reverse2wholeimage(
                    detect_results[0],  # 512x512 crops
                    swap_result_list,  # 512x512 crops
                    detect_results[1],  # mats
                    crop_size,  # 512
                    frame,
                    parsing_model,
                    debug,
                )
            else:  # face not detected
                oriimg, debug_img, final_img = frame, frame, frame

            if debug:
                if final_img.shape[0] < final_img.shape[1]:
                    debug_writer.writeFrame(np.vstack([oriimg, debug_img, final_img]).astype(np.uint8)[..., ::-1])
                else:
                    debug_writer.writeFrame(np.hstack([oriimg, debug_img, final_img]).astype(np.uint8)[..., ::-1])
            writer.writeFrame(final_img[..., ::-1])

            frame_index += 1
            bar.update()


def infer_simswap_and_psfrgan(align_faces, id_vetor, id_vetor_flip, swap_model, use_image_flip):
    """Apply simswap and PSFRGAN"""
    
    swap_result_list = []
    for align_crop in align_faces:
        # Infer SimSwap model
        align_crop_tenor = torch.Tensor(cv2.cvtColor(align_crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].to(DEVICE) / 255
        swap_result = swap_model(None, align_crop_tenor, id_vetor, None, True)[0]
        if use_image_flip:
            swap_result_flip = swap_model(None, torch.flip(align_crop_tenor, (3,)), id_vetor_flip, None, True)[0]
            # Merge origin and flip result
            swap_result = (swap_result + torch.flip(swap_result_flip, (2,))) / 2
        swap_result = swap_result.cpu().detach().numpy().transpose((1, 2, 0))  # RGB
        swap_result = swap_result[..., ::-1]

        swap_result = (swap_result * 255).astype(np.uint8)
        swap_result_list.append(swap_result)
    return swap_result_list


def encode_segmentation_rgb(parse):
    face_part_ids = [1, 2, 3, 6, 7, 8, 9, 10, 12, 13]
    #     mouth_id = 11
    #     lip = 12, 13
    #     l_eye =  4
    #     r_eye = 5
    face_map = np.zeros([parse.shape[0], parse.shape[1]])
    for valid_id in face_part_ids:
        valid_index = np.where(parse == valid_id)
        face_map[valid_index] = 1
    return face_map


class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)
        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()
        return x, mask


def merger(src_face, dst_face, src_mask, dst_mask):
    smooth_mask_net_top = SoftErosion(kernel_size=17, threshold=0.9, iterations=20).to(DEVICE)
    smooth_mask_net_down = SoftErosion(kernel_size=5, threshold=0.9, iterations=7).to(DEVICE)

    src_mask = torch.from_numpy(src_mask).to(DEVICE)
    dst_mask = torch.from_numpy(dst_mask).to(DEVICE)

    mask_top = smooth_mask_net_top(src_mask[:300][None, None])[0].squeeze_() # лоб
    mask_mid = smooth_mask_net_top(dst_mask[150:300][None, None])[0].squeeze_() # глаза до середины носа
    mask_down = smooth_mask_net_down(dst_mask[200:][None, None])[0].squeeze_() # ниже середины носа

    comb_mask = torch.vstack([mask_top[:200], mask_mid[50:mask_mid.shape[0] - 44], mask_down[56:]])
    comb_mask = smooth_mask_net_down(comb_mask[None, None])[0].squeeze_()
    comb_mask = comb_mask.cpu().numpy()[..., None]

    index_mask = comb_mask > 0
    index_mask = index_mask[..., 0]
    src_face[index_mask] = ((dst_face * comb_mask) + (src_face * (1 - comb_mask)))[index_mask]
    result = src_face[:, :, ::-1]
    return result


def reverse2wholeimage(align_faces, swaped_imgs, mats, crop_size, oriimg, parsing_model, debug):
    """Insert swap face back to frame"""

    target_image_list = []
    debug_target_image_list = []
    img_mask_list = []
    for idx, (swaped_img, mat, source_img) in enumerate(zip(swaped_imgs, mats, align_faces)):
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        swaped_img = cv2.cvtColor(swaped_img, cv2.COLOR_BGR2RGB)

        transforms = parsing_model.get_transforms()
        # parse source face
        out = parsing_model(transforms(source_img)[None].to(DEVICE))
        parsing = out.squeeze(0).detach().cpu().numpy().argmax(0).astype(np.uint8)
        src_mask = encode_segmentation_rgb(parsing)
        # parse swapped face
        out = parsing_model(transforms(swaped_img)[None].to(DEVICE))
        parsing = out.squeeze(0).detach().cpu().numpy().argmax(0).astype(np.uint8)
        dst_mask = encode_segmentation_rgb(parsing)

        # merge source and swap faces
        target_image_parsing = merger(
            source_img,
            swaped_img,
            src_mask,
            dst_mask
        )

        # Insert face back to black frame
        orisize = (oriimg.shape[1], oriimg.shape[0])
        target_image = cv2.warpAffine(
            target_image_parsing,
            mat,
            orisize,
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        )
        target_image_list.append(target_image)

        # Create mask for inserted face
        img_white = np.full((crop_size, crop_size), 255, dtype=float)
        img_white = cv2.warpAffine(img_white, mat, orisize, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        img_white[img_white > 20] = 255
        img_mask = img_white
        kernel = np.ones((40, 40), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        kernel_size = (20, 20)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        img_mask_list.append(img_mask[..., None] / 255)

        if debug:
            debug_target_image = cv2.warpAffine(swaped_img[..., ::-1], mat, orisize,
                                                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            debug_target_image_list.append(debug_target_image)

    # Insert face back to frame by mask
    for img_mask, target_image in zip(img_mask_list, target_image_list):
        final_img = img_mask * target_image + (1 - img_mask) * oriimg
    final_img = final_img.astype(np.uint8)

    # if DEBUG = True
    debug_img = None
    for img_mask, target_image in zip(img_mask_list, debug_target_image_list):
        debug_img = img_mask * target_image + (1 - img_mask) * oriimg
    return oriimg, debug_img, final_img


def load_models():
    opt = TestOptions()
    opt.initialize()
    opt.parser.add_argument('-f')  # dummy arg to avoid bug
    opt = opt.parse()
    opt.crop_size = 512
    opt.which_epoch = 550000
    opt.Arc_path = str(Path().home() / '.cache/dcsimswap/download/arcface_model/arcface_checkpoint.tar') 
    opt.checkpoints_dir = str(Path().home() / '.cache/dcsimswap/download/checkpoints')
    torch.nn.Module.dump_patches = False
    opt.name = '512'
    SWAP_MODEL = create_model(opt)
    SWAP_MODEL.eval()

    PSFRGAN = None
    PARSING_MODEL = BiSeNetModel(DEVICE).to(DEVICE).eval()

    mode = 'ffhq'
    FACE_DETECTION_MODEL = Face_detect_crop(name='antelope', root=str(Path().home() / '.cache/dcsimswap/download/insightface_func/models'))
    FACE_DETECTION_MODEL.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)
    return SWAP_MODEL, PSFRGAN, PARSING_MODEL, FACE_DETECTION_MODEL


def run(image_path, video_path, output_video_path, debug, debug_max_frames, use_image_flip):
    """Get target embedding and call swap function"""
    transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    DEVICE = 'cuda:0'

    swap_model, PSFRGAN, parsing_model, face_detection_model = load_models()

    OUTPUT_VIDEO_PATH_DEBUG = f'{output_video_path}debug.mp4'
    video = cv2.VideoCapture(video_path)
    fps = str(video.get(cv2.CAP_PROP_FPS))
    writer = FFmpegWriter(output_video_path,
                        inputdict={'-r': fps},
                        outputdict={"-vcodec": "h264", "-crf": '19', "-preset": "veryslow", 
                                    "-pix_fmt": "yuv420p", "-movflags": "+faststart"})
    if debug:
        debug_writer = FFmpegWriter(OUTPUT_VIDEO_PATH_DEBUG, 
                                    inputdict={'-r': fps},
                                    outputdict={"-vcodec": "h264", "-crf": '19', "-preset": "veryslow", 
                                                "-pix_fmt": "yuv420p", "-movflags": "+faststart"})
    with torch.no_grad():
        latend_id = torch.zeros((1, 512)).to(DEVICE)
        latend_id_flip = torch.zeros((1, 512)).to(DEVICE)
        for pic_a in image_path:
            img_a_whole = io.imread(pic_a)
            img_a_align_crop, _ = face_detection_model.get(img_a_whole, 512)
            align_crop = cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB)

            align_crop = cv2.resize(align_crop, (112, 112), interpolation=cv2.INTER_LANCZOS4)
            img = transformer_Arcface(align_crop)[None].to(DEVICE)

            latend_id += swap_model.netArc(img)
            if use_image_flip:
                latend_id_flip += swap_model.netArc(torch.flip(img, (3,)))

        latend_id = F.normalize(latend_id, p=2, dim=1)
        latend_id_flip = F.normalize(latend_id_flip, p=2, dim=1)

        video_swap(
            latend_id,
            latend_id_flip,
            debug,
            debug_max_frames,
            face_detection_model,
            video,
            debug_writer,
            writer,
            swap_model,
            use_image_flip,
            parsing_model,
        )
    video.release()
    writer.close()
    if debug:
        debug_writer.close()
        
    del swap_model
    del parsing_model
    del face_detection_model
    torch.cuda.empty_cache()


    tmp_output_path = f"{output_video_path}_{uuid.uuid4().hex}.mp4"
    command = f"ffmpeg -y -i {output_video_path} -i {video_path} -c:v copy -map 0:v:0 -map 1:a:0 {tmp_output_path}"
    subprocess.call(command, shell=True, stdout=None)
    os.rename(tmp_output_path, output_video_path)

    if debug:
        command = f"ffmpeg -y -i {OUTPUT_VIDEO_PATH_DEBUG} -i {video_path} -c:v copy -map 0:v:0 -map 1:a:0 {tmp_output_path}"
        subprocess.call(command, shell=True, stdout=None)
        os.rename(tmp_output_path, OUTPUT_VIDEO_PATH_DEBUG)