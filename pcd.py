import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

align = rs.align(rs.stream.color)
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
pipeline = rs.pipeline()
profile = pipeline.start(config)
#########
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)
preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
for i in range(int(preset_range.max)):
    visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
    print('%02d: %s'%(i,visulpreset))
    if visulpreset == "High Accuracy":
        depth_sensor.set_option(rs.option.visual_preset, i)
        break
# enable higher laser-power for better detection
depth_sensor.set_option(rs.option.laser_power, 180)
# lower the depth unit for better accuracy and shorter distance covered
depth_sensor.set_option(rs.option.depth_units, 0.0005)
###########

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
extrinsic = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]


def convert_rs_frames_to_pointcloud(rs_frames):
    aligned_frames = align.process(rs_frames)
    rs_depth_frame = aligned_frames.get_depth_frame()
    np_depth = np.asanyarray(rs_depth_frame.get_data())
    o3d_depth = o3d.geometry.Image(np_depth)

    rs_color_frame = aligned_frames.get_color_frame()
    np_color = np.asanyarray(rs_color_frame.get_data())
    o3d_color = o3d.geometry.Image(np_color)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth, depth_scale=500, convert_rgb_to_intensity=False)
    #rgbd=o3d.geometry.create_rgbd_image_from_color_and_depth(o3d_color, o3d_depth, depth_scale=1
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic,extrinsic)

    return pcd

rs_frames = pipeline.wait_for_frames()
pcd = convert_rs_frames_to_pointcloud(rs_frames)
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud Visualizer",
                  width=800, height=800)
vis.add_geometry(pcd)
render_opt = vis.get_render_option()
render_opt.point_size = 0.25

while True:
    rs_frames = pipeline.wait_for_frames()
    pcd_new = convert_rs_frames_to_pointcloud(rs_frames)
    pcd.points = pcd_new.points
    pcd.colors = pcd_new.colors
    vis.update_geometry(pcd)
    if vis.poll_events():
        vis.update_renderer()
    else:
        o3d.io.write_point_cloud('1.pcd', pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)
        break

vis.destroy_window()
pipeline.stop()



