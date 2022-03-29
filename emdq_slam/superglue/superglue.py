import torch
from .models.superglue import SuperGlue
from .models.superpoint import SuperPoint
torch.set_grad_enabled(False)
import cv2


class SuperGlueMatcher():
    def __init__(self, device):
        self.device = device
        # Load the SuperPoint and SuperGlue models.
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self.superpoint = SuperPoint(config.get('superpoint', {})).to(device)
        self.superglue = SuperGlue(config.get('superglue', {})).to(device)
        self.last_res = None

    def __call__(self, img, mask=None):

            if not torch.is_tensor(img):
                if img.ndim == 3:
                    # convert to grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = torch.from_numpy(img/255.).float()[None, None].to(self.device)
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(mask).to(self.device)

            # detect and compute
            pred = self.superpoint({'image': img})
            if mask is not None:
                self._mask(pred, mask)
            matches = None
            if self.last_res is not None:
                pred0 = {**pred, **{k + '0': v for k, v in self.last_res.items()}}
                pred1 = {**pred, **{k + '1': v for k, v in pred.items()}}

                # Batch all features
                # We should either have i) one image per batch, or
                # ii) the same number of local features for all images in the batch.
                data = {**pred0, **pred1}

                for k in data:
                    if isinstance(data[k], (list, tuple)):
                        data[k] = torch.stack(data[k])

                # Perform the matching
                res = self.superglue(data, img.shape)
                matches_t, conf = res['matches0'][0].cpu().numpy(), res['matching_scores0'][0].cpu().numpy()
                matches = []
                for i, (m, c) in enumerate(zip(matches_t, conf)):
                    if m > -1:
                        matches.append((i, int(m), 1/(c+0.00001)))
            self.last_res = pred
            kpts1 = pred['keypoints'][0].cpu().numpy()
            return kpts1, matches

    def _mask(self, prediction, mask):
        #ToDo make efficient and check x,y
        ids = prediction['keypoints'][0].long()
        kp_mask = mask[ids[:, 1], ids[:, 0]] == 1
        prediction['keypoints'][0] = prediction['keypoints'][0][kp_mask]
        prediction['scores'] = (prediction['scores'][0][kp_mask],)
        prediction['descriptors'][0] = prediction['descriptors'][0][:,kp_mask]
        return prediction
