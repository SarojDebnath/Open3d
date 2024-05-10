import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
import copy
import clr
import time
clr.AddReference("IRA_UR_SocketCtrl_Prog")
import IRA_UR_SocketCtrl_Prog
robot=IRA_UR_SocketCtrl_Prog.SocketCtrl('192.168.1.251',30002,30020,100,100)
robot.Start()
time.sleep(1)

#TRANSFORMATION MATRIX##########
initial_position = list(robot.ActualPoseCartesianRad)
def euler_to_rotation_matrix(rx, ry, rz):
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(rx),     -np.sin(rx) ],
                    [0,         np.sin(rx),      np.cos(rx)  ]])

    R_y = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])

    R_z = np.array([[np.cos(rz),    -np.sin(rz), 0],
                    [np.sin(rz),     np.cos(rz), 0],
                    [0,            0,           1]])

    return np.dot(R_z, np.dot(R_y, R_x))

def calculate_transformation_matrix(initial_position, final_position):
    
    # Extract initial and final positions
    x1, y1, z1, rx1, ry1, rz1 = initial_position
    x2, y2, z2, rx2, ry2, rz2 = final_position

    # Calculate the translation vector
    translation_vector = np.array([x2 - x1, y2 - y1, z2 - z1])
    rotation_matrix = euler_to_rotation_matrix(rx2 - rx1, ry2 - ry1, rz2 - rz1)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix

#PCD######

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

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def icp(source,target,trans_init):
    #source = o3d.io.read_point_cloud("1.pcd")
    #target = o3d.io.read_point_cloud("2.pcd")
    threshold = 0.02
    
    draw_registration_result(source, target, trans_init)
    #print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
                                                        threshold, trans_init)
    #print(evaluation)

    #print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,o3d.pipelines.registration.TransformationEstimationPointToPoint())
    #print(reg_p2p)
    #print("Transformation is:")
    #print(reg_p2p.transformation)
    #print("")
    draw_registration_result(source, target, reg_p2p.transformation)

    #print("Apply point-to-plane ICP")
    #reg_p2l = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,o3d.pipelines.registration.TransformationEstimationPointToPlane())
   #print(reg_p2l)
   #print("Transformation is:")
   #print(reg_p2l.transformation)
   #print("")
    #draw_registration_result(source, target, reg_p2l.transformation)

rs_frames = pipeline.wait_for_frames()
pcd = convert_rs_frames_to_pointcloud(rs_frames)
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud Visualizer",
                  width=800, height=800)
vis.add_geometry(pcd)
render_opt = vis.get_render_option()
render_opt.point_size = 0.25
rs_frames = pipeline.wait_for_frames()
pcd_init = convert_rs_frames_to_pointcloud(rs_frames)
while True:
    rs_frames = pipeline.wait_for_frames()
    pcd_new = convert_rs_frames_to_pointcloud(rs_frames)
    pcd.points = pcd_new.points
    pcd.colors = pcd_new.colors
    pcd=icp(pcd,pcd_init,calculate_transformation_matrix(initial_position, list(robot.ActualPoseCartesianRad)))
    vis.update_geometry(pcd)
    
    if vis.poll_events():
        vis.update_renderer()
        pcd_init=pcd
        initial_position = list(robot.ActualPoseCartesianRad)
    else:
        #o3d.io.write_point_cloud('2.pcd', pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)
        break

vis.destroy_window()
pipeline.stop()


