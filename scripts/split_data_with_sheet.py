import sys
sys.path.append('../')
import os
import pandas as pd
import shutil
import glob


def main(input_path):
    df = pd.read_excel(os.path.join(input_path, "selection_dataset.ods"), squeeze=True)

    def time2frames(t, fps=60):
        min, sec = t.split(":")
        frames = fps*(60*int(min)+int(sec))
        return frames

    df['start_time'] = df['start_time'].apply(time2frames)
    df['end_time'] = df['end_time'].apply(time2frames)

    df_acc = []
    for d in df['dataset'].unique():
        df_d = df[df.dataset.eq(d)]
        for s in df_d['sequence'].unique():
            df_s = df_d[df_d.sequence.eq(s)]
            start_time = df_s['start_time'].min()
            end_time = df_s['end_time'].max()
            df_acc.append({"dataset": d, "sequence": s, "start_time": start_time, "end_time": end_time})

    df_acc = pd.DataFrame(df_acc)

    for j, row in df_acc.iterrows():
        outpath = os.path.join(input_path, row["dataset"] + f"_{row['sequence']}")
        print(outpath)
        os.makedirs(os.path.join(outpath, 'semantic_predictions'), exist_ok=True)
        os.makedirs(os.path.join(outpath, 'video_frames'), exist_ok=True)
        files = sorted(glob.glob(os.path.join(input_path, row["dataset"], 'semantic_predictions', '*.png')))

        shutil.copy(os.path.join(input_path, row["dataset"], "StereoCalibration.ini"), os.path.join(outpath, "StereoCalibration.ini"))

        with open(os.path.join(input_path, row["dataset"], "groundtruth.txt"), "r") as f:
            lines = f.readlines()
            end = min(row["end_time"], len(lines))
            lines = lines[row["start_time"]: end]
        with open(os.path.join(outpath, "groundtruth.txt"), "w") as f:
            f.writelines(lines)

        for i in range(row["start_time"], row["end_time"]):
            try:
                filename = os.path.basename(files[i])
            except IndexError:
                print(f"{i} > {len(files)}. skip")
                break
            shutil.copy(files[i], os.path.join(outpath, 'semantic_predictions', filename))
            shutil.copy(os.path.join(input_path, row["dataset"],'video_frames', filename),
                        os.path.join(outpath, 'video_frames', filename))
            filename_r = filename.replace("l", "r")
            shutil.copy(os.path.join(input_path, row["dataset"],'video_frames', filename_r),
                        os.path.join(outpath, 'video_frames', filename_r))

        print('finished')


if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='script to extract stereo data')

    parser.add_argument(
        'input',
        type=str,
        help='Path to input folder.'
    )
    args = parser.parse_args()

    main(args.input)
