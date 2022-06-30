import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize
from dataset.preprocess.segmentation_network.decoder import DeepLabV3PlusDecoder
from dataset.preprocess.segmentation_network.pvt import pvt_v2_b2
import segmentation_models_pytorch as smp
from dataset.preprocess.segmentation_network.temporal_attention import DeepLabTAM
import os
from dataset.preprocess.segmentation_network.tmp_scaler import TempScaling


def get_model(checkpoint):
    assert os.path.isfile(checkpoint)
    checkp = torch.load(checkpoint, map_location='cpu')
    num_classes = checkp['config']['model']['num_classes']

    model = smp.DeepLabV3Plus(
        encoder_name='efficientnet-b0',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        encoder_output_stride=16,
        classes=num_classes,
    )

    model.encoder = pvt_v2_b2()
    model.decoder = DeepLabV3PlusDecoder(
        encoder_channels=model.encoder.out_channels,
        out_channels=model.decoder.out_channels,
        atrous_rates=(12, 24, 36),
        output_stride=16)
    if checkp['config']['model']['temporal_aggregation'] == 'TAM':
        model = DeepLabTAM(model, input_size=checkp['config']['data'][0]['train']['img_size'])
        model.load_state_dict(checkp['model_state'])
        model = model.segmentation_model
    elif checkp['config']['model']['temporal_aggregation'] == 'none':
        model.load_state_dict(checkp['model_state'])
    else:
        raise ValueError(f"{checkp['config']['model']['temporal_aggregation']} not supported")
    return model, num_classes


class ToTensor(object):
    def __call__(self, input):
        if torch.is_tensor(input):
            return input.unsqueeze(0).permute(0,3,1,2).float()/255.0
        else:
            return torch.tensor(input).unsqueeze(0).permute(0,3,1,2).float() / 255.0


class SemanticSegmentationModel(nn.Module):
    def __init__(self, checkpoint, device, temperature=1.0, background_prior=1.0):
        super(SemanticSegmentationModel, self).__init__()
        self.model, self.n_classes = get_model(checkpoint)
        self.model.to(device)
        prior = torch.ones(self.n_classes)
        prior[0] = background_prior
        self.calibration = TempScaling(apply_softmax=True, temperature=temperature, prior=prior).to(device)
        self.transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.device = device
        self.model.eval()
        self.eval()
        self.tool_ids = torch.tensor([1, 2, 3, 8, 9, 11, 12], dtype=torch.uint8, device=device) # tool channel indicies

    def get_mask(self, image):
        with torch.no_grad():
            is_numpy = not torch.is_tensor(image)
            image = self.transform(image).to(self.device)
            soft_predictions = self.forward(image)
            mask = self.get_mask_from_prediction(soft_predictions)
            soft_predictions = soft_predictions.squeeze(0)
        if is_numpy:
            mask = mask.cpu().numpy()
            soft_predictions = soft_predictions.cpu().numpy()
        return mask, soft_predictions

    def get_mask_from_prediction(self, predictions):
        predictions = torch.argmax(predictions, dim=-1)
        mask = torch.ones(predictions.shape[:2], dtype=torch.uint8, device=self.device)
        mask[(predictions[..., None] == self.tool_ids).any(-1)] = 0
        return mask

    def segment(self, image):
        with torch.no_grad():
            image = self.transform(image)
            predictions = self.forward(image)
            return predictions

    def forward(self, img):
        out = self.model(img)
        out = self.calibration(out)
        out = out.squeeze(0).permute(1,2,0)
        return out


