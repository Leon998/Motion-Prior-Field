from utils import *
from segpose_net import SegPoseNet
import cv2


def evaluate(data_cfg, weightfile, vidfile, outdir, object_names, intrinsics, vertex,
                         bestCnt, conf_thresh, linemod_index=False, use_gpu=False, gpu_id='0'):
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)

    data_options = read_data_cfg(data_cfg)
    m = SegPoseNet(data_options)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        m.cuda()

    # with open(listfile, 'r') as file:
    #     imglines = file.readlines()
    # capture = cv2.VideoCapture(vidfile)
    capture = cv2.VideoCapture(0)
    i = 0
    while True:
        ret, img = capture.read()
        if not ret:
            break
        outFileName = str(i)

        start = time.time()
        predPose = do_detect(m, img, intrinsics, bestCnt, conf_thresh, use_gpu)
        finish = time.time()

        arch = 'CPU'
        if use_gpu:
            arch = 'GPU'
        print('%s: Predict %d objects in %f seconds (on %s).' % (str(i), len(predPose), (finish - start), arch))
        # save_predictions(outFileName, predPose, object_names, outdir)

        # visualize predictions
        vis_start = time.time()
        visImg = visualize_predictions(predPose, img, vertex, intrinsics)
        cv2.imshow('vis', visImg)
        cv2.waitKey(1)
        # cv2.imwrite(outdir + '/' + outFileName + '.jpg', visImg)
        vis_finish = time.time()
        print('%s: Visualization in %f seconds.' % (str(i), (vis_finish - vis_start)))
        i += 1


if __name__ == '__main__':
    use_gpu = True
    # use_gpu = False
    #
    # dataset = 'Occluded-LINEMOD'
    # outdir = './Occluded-LINEMOD-Out'
    #
    dataset = 'YCB-Video'
    outdir = 'pose_estimation/Real-YCB-Video-Out'
    # intrinsics of YCB-VIDEO dataset
    # k_ycbvideo = np.array([[1.06677800e+03, 0.00000000e+00, 3.12986900e+02],
    #                        [0.00000000e+00, 1.06748700e+03, 2.41310900e+02],
    #                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    k_ycbvideo = np.array([[0.338569200e+03, 0.00000000e+00, 3.17869800e+02],
                           [0.00000000e+00, 0.338800400e+03, 2.45317800e+02],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # 21 objects for YCB-Video dataset
    object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                             '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                             '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                             '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
    vertex_ycbvideo = np.load('pose_estimation/data/YCB-Video/YCB_vertex.npy')
    evaluate('pose_estimation/data/data-YCB.cfg', 'pose_estimation/model/ycb-video.pth', 
             'pose_estimation/videos/mug523.mp4', outdir, object_names_ycbvideo,
             k_ycbvideo, vertex_ycbvideo, bestCnt=10, conf_thresh=0.3, use_gpu=use_gpu)
