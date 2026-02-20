import os
import argparse
import shutil
import json
import pymeshlab
import open3d as o3d
import numpy as np

# group_size = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default='')
    parser.add_argument('--end', type=int, default='')
    parser.add_argument('--cuda', type=int, default='')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--sh', type=str, default='0')
    parser.add_argument('--interval', type=str, default='')
    parser.add_argument('--group_size', type=str, default='')
    parser.add_argument('--resolution', type=int, default=2)
    parser.add_argument('--aabb_scale', type=int, default=2, help='NeRF AABB scale (1=unit cube). Use 2 or 4 for ActorsHQ to avoid head clipping.')
    parser.add_argument('--marching_cubes_res', type=int, default=None,
                        help='NeuS2 marching cubes grid resolution. Default: 500*aabb_scale (e.g. 1000 when aabb_scale=2) for consistent mesh density.')
    args = parser.parse_args()

    print(args.start, args.end)

    # os.system("conda activate torch")
    card_id = args.cuda
    data_root_path = args.data
    output_path = args.output
    sh = args.sh
    interval = int(args.interval)
    group_size = int(args.group_size)
    resolution_scale = int(args.resolution)
    aabb_scale = int(args.aabb_scale)
    marching_cubes_res = int(args.marching_cubes_res) if args.marching_cubes_res is not None else (500 * aabb_scale)

    # neus2_meshlab_filter_path = os.path.join(data_root_path, "luoxi_filter.mlx")

    neus2_output_path = os.path.join(output_path, "neus2_output")
    if not os.path.exists(neus2_output_path):
        os.makedirs(neus2_output_path)

    gaussian_output_path = os.path.join(output_path, "checkpoint")

    for i in range(args.start, args.end, group_size * interval):
        group_start = i
        group_end = min(i + group_size * interval, args.end) - 1
        print(group_start, group_end)
        
        frame_path = os.path.join(data_root_path, str(i))
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        frame_neus2_output_path = os.path.join(neus2_output_path, str(i))
        if not os.path.exists(frame_neus2_output_path):
            os.makedirs(frame_neus2_output_path)
        frame_neus2_ckpt_output_path = os.path.join(frame_neus2_output_path, "frame.msgpack")
        frame_neus2_mesh_output_path = os.path.join(frame_neus2_output_path, "points3d.obj")
        
        """NeuS2"""
        # neus2 command (marching_cubes_res scaled by aabb_scale for consistent mesh density)
        script_path = "scripts/run.py"
        neus2_command = f"cd ../../external/NeuS2_K && CUDA_VISIBLE_DEVICES={card_id} python {script_path} --scene {frame_path} --name neus --mode nerf --save_snapshot {frame_neus2_ckpt_output_path} --save_mesh --save_mesh_path {frame_neus2_mesh_output_path} --marching_cubes_res {marching_cubes_res} && cd ../../scripts/test"
        os.system(neus2_command)
        delete_neus2_output_path = os.path.join(frame_path, "output")
        shutil.rmtree(delete_neus2_output_path)

        # revert axis
        mesh1 = o3d.io.read_triangle_mesh(frame_neus2_mesh_output_path)
        vertices = np.asarray(mesh1.vertices)
        vertices = vertices[:,[2,0,1]]
        mesh1.vertices = o3d.utility.Vector3dVector(vertices)
        o3d.io.write_triangle_mesh(frame_neus2_mesh_output_path, mesh1)

        # use pymeshlab to convert obj to point cloud
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(frame_neus2_mesh_output_path)
        # ms.load_filter_script(neus2_meshlab_filter_path)
        # ms.apply_filter_script()
        ms.generate_simplified_point_cloud(samplenum = 100000) 
        frame_points3d_output_path = os.path.join(frame_path, "points3d.ply")
        ms.save_current_mesh(frame_points3d_output_path, binary = True, save_vertex_normal = False)