import sys
sys.path.append('../')
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
from dataset.dataset_utils import get_data
from core.pose.pose_net import PoseNet


def main(args):
    device = torch.device('cuda')

    WARM_UP_CYCLES = 20  # number of cycles to warm-up GPU
    MEASUREMENT_CYCLES = 100  # perform inference on 1000 frames and average inference time

    # load data and model
    dataset, calib = get_data(args.input, (640, 512), force_stereo=True, rect_mode='conventional')
    checkp = torch.load('../trained/poseNet_2xf8up4b.pth')
    checkp['config']['model']['image_shape'] = (640, 512)
    checkp['config']['model']['amp'] = True if args.amp else False
    checkp['config']['model']['lbgfs_iters'] = 20
    model = PoseNet(checkp['config']['model'])
    new_state_dict = OrderedDict()
    state_dict = checkp['state_dict']
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.to(device)
    intrinsics = torch.tensor(calib['intrinsics']['left']).to(device).unsqueeze(0).float()
    baseline = torch.tensor(calib['bf']).unsqueeze(0).float().to(intrinsics.device)/250.0
    loader = DataLoader(dataset, num_workers=0, pin_memory=True)
    if args.with_seg:
        print("profile with Segmentation Model")
        seg_model = SemanticSegmentationModel('../dataset/preprocess/segmentation_network/trained/deepLabv3plus_trained_intuitive.pth',
                                              device, (640, 512))
    print(' Profiling Inference')
    print(' running on: ', torch.cuda.get_device_properties(0))

    with torch.no_grad():
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = np.zeros((MEASUREMENT_CYCLES, 1))
        # load data into GPU memory and warm-up
        data_list = []
        for ii, data in enumerate(loader):  # for each of the batches
            data_list.append([d.to(device) for d in data[:-2]])
            if ii == 1:
                break

        def infer(data, last_img, last_depth, last_flow, last_valid):
            limg, rimg, tool_mask = data
            tool_mask = tool_mask.bool()
            with torch.inference_mode():
                if args.with_seg:
                    tool_mask = seg_model.get_mask(limg/255.0)[0].to(torch.bool)
            # pose estimation
            model.infer(last_img, limg, intrinsics, baseline, depth1=last_depth, image2r=rimg,
                  mask1=last_valid, mask2=tool_mask, stereo_flow1=last_flow)

        limg, rimg, tool_mask = data_list[0]
        tool_mask = tool_mask.bool()
        depth, flow, valid = model.flow2depth(limg, rimg, baseline)
        valid &= tool_mask
        last_data = (limg, depth, flow, valid)

        # GPU-WARM-UP
        for _ in range(WARM_UP_CYCLES):
            infer(data_list[1], *last_data)

        # MEASURE PERFORMANCE

        for rep in range(MEASUREMENT_CYCLES):
            starter.record()
            infer(data_list[1], *last_data)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    average_time = np.sum(timings) / MEASUREMENT_CYCLES
    std_time = np.std(timings)
    print('Inference Sync: Average Time per Frame: {0:0.2f}+/-{2:0.2f} ms, Frames per Second: {1:.2f}'.format(
        average_time, 1 / average_time * 1000, std_time))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='script to run Raft Pose SLAM')

    parser.add_argument(
        'input',
        type=str,
        help='Path to input folder.'
    )
    parser.add_argument(
        '--with_seg',
        action="store_true",
        help='include segmentation in profiling.'
    )
    parser.add_argument(
        '--iters',
        type=int,
        default=12,
        help='number of RAFT iterations.'
    )
    parser.add_argument(
        '--amp',
        action="store_true",
        help='use automatic mixed precision.'
    )
    args = parser.parse_args()

    main(args)
