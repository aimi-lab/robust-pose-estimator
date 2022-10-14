try:
    import matplotlib.pyplot as plt
    SHOW = True
except ImportError:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    SHOW = False
import os


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

    def __call__(self, frame, synth_frame, idx:int=0):
        # plot input frame and synthesized frame
        fig, ax = plt.subplots(2, 3, num=1, clear=True, figsize=(10,8))
        img_t, _,_, depth_t, *_, conf_t = frame.to_numpy()
        img, _,_, depth, *_, conf = synth_frame.to_numpy()
        ax[0,0].imshow(img_t)
        ax[0, 0].axis('off')
        ax[0, 0].set_title('I_t')
        ax[0, 1].imshow(depth_t, vmin=0, vmax=1)
        ax[0, 1].axis('off')
        ax[0, 1].set_title('depth_t')
        ax[0, 2].imshow(conf_t*conf, vmin=0, vmax=1)
        ax[0, 2].axis('off')
        ax[0, 2].set_title('weights_t*weights_t-1')

        ax[1, 0].imshow(img)
        ax[1, 0].axis('off')
        ax[1, 0].set_title('I_t-1')
        ax[1, 1].imshow(depth, vmin=0, vmax=1)
        ax[1, 1].axis('off')
        ax[1, 1].set_title('depth_t-1')
        ax[1, 2].imshow(conf, vmin=0, vmax=1)
        ax[1, 2].axis('off')
        ax[1, 2].set_title('weights_t-1')
        plt.tight_layout()
        if not self.blocking & SHOW:
            plt.draw()
        if self.outpath is not None:
            plt.savefig(os.path.join(self.outpath, f'vis_{idx:06d}.png'))
        if self.blocking & SHOW:
            plt.show()
        if not self.blocking & SHOW:
            plt.pause(0.0001)
        if not SHOW:
            plt.close()

