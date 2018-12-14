import os

def count_frames(path):
    frame_count=len(os.listdir(path))
    assert frame_count > 0, \
        "VideoIter:: Video: `{}' has no frames".format(path)
    return frame_count

def get_file_list(video_prefix,txt_list,cached_info_path):

    assert os.path.exists(video_prefix), "VideoIter:: failed to locate: `{}'".format(video_prefix)
    assert os.path.exists(txt_list), "VideoIter:: failed to locate: `{}'".format(txt_list)

    video_list=[]
    # building dataset
    with open(txt_list) as f:
        lines = f.read().splitlines()
        print("VideoIter:: found {} videos in `{}'".format(len(lines), txt_list))
        for i, line in enumerate(lines):
            v_id, label, video_subpath = line.split()
            video_subpath=video_subpath[:-4]
            video_path = os.path.join(video_prefix, video_subpath)
            if not os.path.exists(video_path):
                print("VideoIter:: >> cannot locate `{}'".format(video_path))
                continue

            frame_count = count_frames(video_path)
            info = [video_subpath, frame_count,int(label)]
            video_list.append(info)

        with open(cached_info_path, 'w') as f:
            for i, video in enumerate(video_list):
                if i > 0:
                    f.write("\n")
                f.write("{:s}\t\t{:d}\t{:d}".format(video[0], video[1],video[2]))
    

if __name__ == '__main__':
    get_file_list('./data','./list_cvt/trainlist01.txt','train_list.txt')
    get_file_list('./data','./list_cvt/testlist01.txt','test_list.txt')