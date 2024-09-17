import click
import torch
from os.path import join
from loc_ndf.utils import registration, vis, utils
from tqdm import tqdm
from pathlib import Path
import numpy as np


@click.command()
# Add your options here
@click.argument('checkpoints',
                nargs=-1,
                required=True)
@click.option('--num_voxels',
              '-v',
              type=int,
              default=400,
              required=True)
@click.option('--threshold',
              '-t',
              type=float,
              default=0.01,
              required=False)
@click.option('--visualize', '-vis', is_flag=True, show_default=True, default=False)
@click.option('--do_test', '-test', is_flag=True, show_default=True, default=False)
@click.option('--dataset', required=True)
def main(checkpoints, num_voxels, threshold, visualize, do_test, dataset):

    if do_test:
        folder = join(utils.DATA_DIR,f"{dataset}/TestData/ColumbiaPark/2018-10-11")
        start_idx = 5280
        num_scans = 700
        prefix = 'test'
    else:
        folder = join(utils.DATA_DIR,
                      f"{dataset}/TrainData/ColumbiaPark/2018-10-03")
        start_idx = 6880
        num_scans = 800
        prefix = 'validation'

    tracker = registration.PoseTracker(
        checkpoints=checkpoints,
        test_folder=folder,
        start_idx=start_idx,
        GM_k=0.3,
        max_dist=75,
        num_points=-1,
        nv=num_voxels, 
        threshold=threshold)

    if visualize:
        visulizer = vis.Visualizer(tracker)
        visulizer.run()
    else:
        gt_poses = []
        est_poses = []
        for i in tqdm(range(num_scans)):
            est, gt, _ = tracker.register_next()
            est_poses.append(est)
            gt_poses.append(gt)
        gt_poses = torch.stack(gt_poses)
        est_poses = torch.stack(est_poses)

        dt, dr = registration.pose_error(
            gt_poses, est_poses)
        memory = tracker.get_memory()

        print('Final errors')
        print('AE translation / rotation')
        print(dt, dr)
        print(f"memory {memory:.3f}MB")

        results_dir = Path(utils.RESULTS_DIR) / dataset
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / f'{prefix}_odom_error.txt', 'w') as f:
            results = f"# dt [m], dr [deg], memory [MB]\n{dt} {dr} {memory}"
            f.write(results)
        np.savetxt(results_dir / 'gt_posts.txt', gt_poses.cpu().numpy().reshape((gt_poses.shape[0], -1)))
        np.savetxt(results_dir / 'est_poses.txt', est_poses.cpu().numpy().reshape((est_poses.shape[0], -1)))

if __name__ == "__main__":
    main()
