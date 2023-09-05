try:
    import matplotlib.pyplot as plt
    SHOW = True
except ImportError:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    SHOW = False
import os
from torchvision.utils import flow_to_image


class Viewer2D(object):
    def __init__(self, outpath=None, blocking=False):
        super().__init__()
        self.blocking = blocking
        self.outpath = os.path.join(outpath, 'imgs') if outpath is not None else None

        if not blocking & SHOW:
            plt.ion()
            plt.show()

        if self.outpath is not None:
            os.makedirs(self.outpath, exist_ok=True)

    def __call__(self, frame, weights, flow, idx:int=0):
        # plot input frame and synthesized frame
        fig, ax = plt.subplots(1, 5, num=1, clear=True, figsize=(10,8))
        img_t, _, depth_t, *_ = frame.to_numpy()
        flow_rgb = flow_to_image(flow.squeeze().cpu()).permute(1,2,0).numpy()
        ax[0].imshow(img_t)
        ax[0].axis('off')
        ax[0].set_title('I_t')
        ax[1].imshow(depth_t, vmin=0)
        ax[1].axis('off')
        ax[1].set_title('depth_t')
        ax[2].imshow(flow_rgb)
        ax[2].axis('off')
        ax[2].set_title('flow')
        ax[3].imshow(weights[0].squeeze().cpu().numpy(), vmin=0)
        ax[3].axis('off')
        ax[3].set_title('w_2d')
        ax[4].imshow(weights[1].squeeze().cpu().numpy(), vmin=0)
        ax[4].axis('off')
        ax[4].set_title('w_3d')

        plt.tight_layout()
        if not self.blocking & SHOW:
            plt.draw()
        if self.outpath is not None:
            plt.savefig(os.path.join(self.outpath, f'vis_{idx:06d}.png'), dpi=300)
        if self.blocking & SHOW:
            plt.show()
        if not self.blocking & SHOW:
            plt.pause(0.0001)
        if not SHOW:
            plt.close()

