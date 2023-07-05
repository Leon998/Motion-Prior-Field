from utils import *
from segpose_net import SegPoseNet
import cv2


def pred_pose(m, raw_img, object_names, target_object, intrinsics, vertex,
                         bestCnt, conf_thresh, linemod_index=False, use_gpu=False, gpu_id='0'):
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        m.cuda()

    start = time.time()
    predPose = do_detect(m, raw_img, intrinsics, bestCnt, conf_thresh, use_gpu)
    finish = time.time()

    arch = 'CPU'
    if use_gpu:
        arch = 'GPU'
    print('Predict %d objects in %f seconds (on %s).' % (len(predPose), (finish - start), arch))
    # visualize predictions
    vis_start = time.time()
    visImg = visualize_predictions(predPose, raw_img, vertex, intrinsics)
    cv2.imshow('vis', visImg)
    cv2.waitKey(1)
    # cv2.imwrite(outdir + '/' + outFileName + '.jpg', visImg)
    vis_finish = time.time()
    print('Visualization in %f seconds.' % (vis_finish - vis_start))
    if predPose:
        pose, detect_flag = select_target_obj_pose(predPose, object_names, target_object)
    else:
        detect_flag = 0
        pose = 0
    return pose, detect_flag

if __name__ == '__main__':
    use_gpu = True
    dataset = 'YCB-Video'
    # intrinsics
    k_ycbvideo = np.array([[0.338856700e+03, 0.00000000e+00, 3.12340100e+02],
                               [0.00000000e+00, 0.339111500e+03, 2.46983900e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # 21 objects for YCB-Video dataset
    object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                                 '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                                 '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                                 '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
    target_object = '025_mug'
    vertex_ycbvideo = np.load('pose_estimation/data/YCB-Video/YCB_vertex.npy')
    raw_img = '/home/shixu/My_env/Motion-Prior-Field/pose_estimation/images/000.jpg'
    raw_img = cv2.imread(raw_img)
    data_cfg = 'pose_estimation/data/data-YCB.cfg'
    weightfile = 'pose_estimation/model/ycb-video.pth'
    data_options = read_data_cfg(data_cfg)
    m = SegPoseNet(data_options)
    m.load_weights(weightfile)
    pred_pose(m, raw_img, object_names_ycbvideo, target_object, k_ycbvideo,
              vertex_ycbvideo, bestCnt=10, conf_thresh=0.3, use_gpu=use_gpu)