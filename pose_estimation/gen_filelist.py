import os


def collect_occluded_linemod_testlist(rootpath, outname):
    path = rootpath + 'RGB-D/rgb_noseg/'
    imgs = [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    imgs.sort()
    # write sets
    allf = open(outname, 'w')
    for i in imgs:
        allf.write(path + i + '\n')


def collect_ycb_testlist(rootpath, outfile):
    testListFile = rootpath + '/image_sets/keyframe.txt'
    with open(testListFile, 'r') as file:
        testlines = file.readlines()
    with open(outfile, 'w') as file:
        for l in testlines:
            file.write(rootpath + 'data/' + l.rstrip() + '-color.png\n')


def collect_real_ycblist(rootpath, outfile):
    files = os.listdir(rootpath)
    files.sort()
    with open(outfile, 'w') as file:
        for l in files:
            file.write(rootpath + l + '\n')


if __name__ == '__main__':
    # modify the path according to your real path of the Occluded-LINEMOD and YCB-Video dataset

    # occluded_linemod_path = '/data/OcclusionChallengeICCV2015/'
    # collect_occluded_linemod_testlist(occluded_linemod_path, './occluded-linemod-testlist.txt')

    # ycb_video_path = '/home/shixu/My_env/Dataset/YCB_Video_Dataset/'
    # collect_ycb_testlist(ycb_video_path, './ycb-video-testlist.txt')

    real_ycb_path = '/home/shixu/My_env/Motion-Prior-Field/pose_estimation/images/'
    collect_real_ycblist(real_ycb_path, '/home/shixu/My_env/Motion-Prior-Field/pose_estimation/real_ycb.txt')
