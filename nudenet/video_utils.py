import cv2
import os
import logging

# logging.basicConfig(level=logging.DEBUG)

from skimage import metrics as skimage_metrics


def is_similar_frame(f1, f2, resize_to=(64, 64), thresh=0.5, return_score=False):
    thresh = float(os.getenv("FRAME_SIMILARITY_THRESH", thresh))

    if f1 is None or f2 is None:
        return False

    if isinstance(f1, str) and os.path.exists(f1):
        try:
            f1 = cv2.imread(f1)
        except Exception as ex:
            logging.exception(ex, exc_info=True)
            return False

    if isinstance(f2, str) and os.path.exists(f2):
        try:
            f2 = cv2.imread(f2)
        except Exception as ex:
            logging.exception(ex, exc_info=True)
            return False

    if resize_to:
        f1 = cv2.resize(f1, resize_to)
        f2 = cv2.resize(f2, resize_to)

    if len(f1.shape) == 3:
        f1 = f1[:, :, 0]

    if len(f2.shape) == 3:
        f2 = f2[:, :, 0]

    score = skimage_metrics.structural_similarity(f1, f2, multichannel=False)

    if return_score:
        return score

    if score >= thresh:
        return True

    return False


def get_interest_frames_from_video(
    video_path,
    frame_similarity_threshold=0.5,
    similarity_context_n_frames=3,
    skip_n_frames=0.5,
    output_frames_to_dir=None,
):
    skip_n_frames = float(os.getenv("SKIP_N_FRAMES", skip_n_frames))

    important_frames = []
    fps = 0
    video_length = 0

    try:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if skip_n_frames < 1:
            skip_n_frames = int(skip_n_frames * fps)
            logging.info(f"skip_n_frames: {skip_n_frames}")

        video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_i in range(length + 1):
            read_flag, current_frame = video.read()

            if not read_flag:
                break

            if skip_n_frames > 0:
                if frame_i % skip_n_frames != 0:
                    continue

            frame_i += 1

            found_similar = False
            for context_frame_i, context_frame in reversed(
                important_frames[-1 * similarity_context_n_frames :]
            ):
                if is_similar_frame(
                    context_frame, current_frame, thresh=frame_similarity_threshold
                ):
                    logging.debug(f"{frame_i} is similar to {context_frame_i}")
                    found_similar = True
                    break

            if not found_similar:
                logging.debug(f"{frame_i} is added to important frames")
                important_frames.append((frame_i, current_frame))
                if output_frames_to_dir:
                    if not os.path.exists(output_frames_to_dir):
                        os.mkdir(output_frames_to_dir)

                    output_frames_to_dir = output_frames_to_dir.rstrip("/")
                    cv2.imwrite(
                        f"{output_frames_to_dir}/{str(frame_i).zfill(10)}.png",
                        current_frame,
                    )

        logging.info(
            f"{len(important_frames)} important frames will be processed from {video_path} of length {length}"
        )

    except Exception as ex:
        logging.exception(ex, exc_info=True)

    return (
        [i[0] for i in important_frames],
        [i[1] for i in important_frames],
        fps,
        video_length,
    )


if __name__ == "__main__":
    import sys

    imp_frames = get_interest_frames_from_video(
        sys.argv[1], output_frames_to_dir="./frames/"
    )
    print([i[0] for i in imp_frames])
