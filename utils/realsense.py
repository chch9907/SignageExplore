import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import argparse
# https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb
# https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
class RealSenseCameraBase:
    def __init__(self, width=640, height=480, fps=30, name='D455'):
        self.name = name
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps) # depth frames configuration
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps) # color frames configuration
        ctx = rs.context()
        devices = ctx.query_devices()
        for d in devices:
            if d.get_info(rs.camera_info.name) == f'Intel RealSense {name}':
                config.enable_device(d.get_info(rs.camera_info.serial_number))
                break

        self.profile = self.pipeline.start(config)
        # Get data scale from the device and convert to meters
        device = self.profile.get_device()
        
        
        #* manually set exposure
        # exposure_value = 100
        # color_sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1] # index '1' means color rather than depth, or using first_color_sensor()
        # color_sensor.set_option(rs.option.enable_auto_exposure, False)
        # color_sensor.set_option(rs.option.exposure, exposure_value)
        
        # Reset to the default values: 156 for exposure and 64 for gain.
        # device.hardware_reset()
        
        self.depth_scale = device.first_depth_sensor().get_depth_scale()  # 0.001
        self.align = rs.align(rs.stream.color)
        
        self.threshold_filter = rs.threshold_filter()  # default 0.15 ~ 4m
        # self.threshold_filter.set_option(rs.option.max_distance,max_threshold)
        # self.threshold_filter.set_option(rs.option.min_distance,min_threshold)
        self.decimation = rs.decimation_filter()
        # self.spatial = rs.spatial_filter() # time wasting: https://github.com/IntelRealSense/librealsense/issues/10995#issuecomment-1278909080
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        
        # Skip 5 first frames to give the Auto-Exposure time to adjust
        for _ in range(5):
            frames = self.pipeline.wait_for_frames(10000)
            filtered_frames = self.depth_filter(frames)
        self.init_intrinsics()

    def get_intrinsics(self, frame):
        return frame.profile.as_video_stream_profile().intrinsics

    def init_intrinsics(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        self.color_intrinsics = self.get_intrinsics(color_frame)
        self.depth_intrinsics = self.get_intrinsics(depth_frame)

        self.camera_parameters = {
            'fx': self.color_intrinsics.fx,
            'fy': self.color_intrinsics.fy,
            'ppx': self.color_intrinsics.ppx,
            'ppy': self.color_intrinsics.ppy,
            'height': self.color_intrinsics.height,
            'width': self.color_intrinsics.width,
            'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale()
        }

    def depth_filter(self, frame):
        frame = self.threshold_filter.process(frame)
        # frame = self.decimation.process(frame)
        frame = self.depth_to_disparity.process(frame)
        # frame = self.spatial.process(frame)  # take much time
        frame = self.temporal.process(frame) 
        frame = self.disparity_to_depth.process(frame)
        frame = self.hole_filling.process(frame)
        return frame
        
    def get_image(self):
        frames = self.pipeline.wait_for_frames()
        # filtered_frames = self.depth_filter(frames)  # improve depth frame performance
        aligned_frames = self.align.process(frames) #.as_frameset()
        color_frame = aligned_frames.get_color_frame()
        color_img = np.asanyarray(color_frame.get_data())
        
        depth_frame = aligned_frames.get_depth_frame()
        depth_frame = self.depth_filter(depth_frame)  # improve depth frame performance
        depth_frame = depth_frame.as_depth_frame()  # cast it as depth frame after filter processing
        depth_img = np.asanyarray(depth_frame.get_data()) # by default 16-bit
        
        return color_img, depth_img, depth_frame

    def get_3d_in_camera_frame_from_2d(self, coord_2d, depth_frame):
        distance = depth_frame.get_distance(*coord_2d)
        print('distance:', distance)
        world_coordinate = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, coord_2d, distance)
        return world_coordinate  # 2D to 3D

    def stop(self,):
        self.pipeline.stop()


class RealSenseCamera(RealSenseCameraBase):
    def __init__(self, width=640, height=480, fps=30, name='D455'):
    # def __init__(self, width=1280, height=720, fps=30, name='D455'):
        super().__init__(width, height, fps, name)
        cv.namedWindow(self.name, cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback(self.name, self.click)

    def click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print('clicked picture coord: ', x, y)
            color_img, depth_img, depth_frame = self.get_image()
            print('get depth: not implemented.')
            # xyz = self.get_3d_in_camera_frame_from_2d((x, y), depth_frame)
            # print('clicked camera coord: ', xyz)

    def get_image(self, show=False):
        color_img, depth_img, depth_frame = super().get_image()
        if show:
            cv.imshow(self.name, color_img)
            cv.waitKey(1)
            print(color_img.shape)
        return color_img, depth_img, depth_frame

class RealSenseCameraDepth(RealSenseCamera):
    def get_image(self):
        color_img, depth_img, depth_frame = super().get_image()
        real_depth = depth_img * self.depth_scale  # convert depth image to depth matrix
        # depth_img_8bit = cv.convertScaleAbs(depth_img, alpha=0.05) # convert to 8-bit
        # depth_colormap = cv.applyColorMap(depth_img_8bit, cv.COLORMAP_JET)
        # depth_img_3d = np.dstack([depth_img_8bit]*3)
        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        h, w = depth_colormap.shape[:2]
        color_img = cv.resize(color_img, (w, h))
        color_depth = np.hstack([color_img, depth_colormap])
        
        return color_img, real_depth, color_depth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='D455')
    parser.add_argument('--auto_show', default=True, type=bool)
    parser.add_argument('--show_depth', action='store_true')
    args = parser.parse_args()
    
    auto_show = args.auto_show  # True
    show_depth = args.show_depth  # False
    device_name = args.name
    if not auto_show:
        cam = RealSenseCameraBase(name=device_name)
    else:
        if not show_depth:
            cam = RealSenseCamera(name=device_name)
        else:
            cam = RealSenseCameraDepth(name=device_name)

    if not auto_show:
        while True:
            color_img, depth_img, depth_frame = cam.get_image(show=False)
            
            cv.imshow(cam.name, color_img)
            if cv.waitKey(1) & 0xFF==ord('q'):
                break
    else:
        while True:
            color_img, depth_img, color_depth = cam.get_image()
            cv.imshow(cam.name, color_depth)
            if cv.waitKey(1) & 0xFF==ord('q'):
                break

    cv.destroyAllWindows()