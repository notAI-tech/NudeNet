import cv2
import logging

from skimage import metrics as skimage_metrics

def is_similar_frame(f1, f2, resize_to=(64, 64), thresh=0.8):
    if f1 is None or f2 is None: return False

    if resize_to:
        f1 = cv2.resize(f1, resize_to)
        f2 = cv2.resize(f2, resize_to)

    if len(f1.shape) == 3:
        f1 = f1[:, :, 0]

    if len(f2.shape) == 3:
        f2 = f2[:, :, 0]

    score = skimage_metrics.structural_similarity(f1, f2, multichannel=False)

    if score >= thresh:
        return True

    return False

def get_interest_frames_from_video(video_path, frame_similarity_threshold=0.8, similarity_context_n_frames=3, skip_n_frames=0.5):
    important_frames = []

    try:
        video = cv2.VideoCapture(video_path)
        if skip_n_frames < 1:
            fps = video.get(cv2.CAP_PROP_FPS)
            skip_n_frames = int(skip_n_frames * fps)
            logging.info(f'skip_n_frames: {skip_n_frames}')

        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_i in range(length + 1):
            if frame_i % skip_n_frames != 0:
                continue

            frame_i += 1
            read_flag, current_frame = video.read()

            if not read_flag:
                break

            found_similar = False
            for context_frame_i, context_frame in important_frames[-1 * similarity_context_n_frames:]:
                if is_similar_frame(context_frame, current_frame, thresh=frame_similarity_threshold):
                    logging.debug(f'{frame_i} is similar to {context_frame_i}')
                    found_similar = True
                    break

            if not found_similar:
                logging.debug(f'{frame_i} is added to important frames')
                important_frames.append((frame_i, current_frame))
            
        logging.info(f'{len(important_frames)} important frames will be processed from {video_path} of length {length}')
    
    except Exception as ex:
        logging.exception(ex, exc_info=True)

    return important_frames


if __name__ == '__main__':
    import sys
    imp_frames = get_interest_frames_from_video(sys.argv[1])
    print([i[0] for i in imp_frames])

            


