import cv2
import os
import zfa.core.video_utils as vu


def extract_frames_from_video(
    video_path, output_folder, shortest_side_size=None, resize_shape=None
):
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        if shortest_side_size is not None:
            # sanity check
            assert frame.shape[0] == cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            assert frame.shape[1] == cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame = vu.shortest_side_resize(
                image=frame, shortest_side_size=shortest_side_size
            )
        else:
            frame = cv2.resize(frame, resize_shape)

        # Construct the output path for the frame
        frame_file_path = os.path.join(output_folder, f"frame_{frame_count:010d}.jpg")

        # Save the frame as a JPEG file
        cv2.imwrite(frame_file_path, frame)

        if frame_count % 500 == 0:
            print(f"Extracted frame {frame_count}")

        # Move to the next frame
        frame_count += 1

    # Release the video capture object
    cap.release()

    print("Extraction complete.")


if __name__ == "__main__":

    # default height and width to resize RGB frames to
    OUT_HEIGHT_WIDTH = (224, 224)

    video_path = "/om2/group/yanglab/zfa/Wild zebrafish Danio rerio in India.mp4"  # Replace with the path to your .mp4 video file
    output_folder = "/om2/group/yanglab/zfa/Frames/"  # Replace with the folder where you want to save the frames
    extract_frames_from_video(
        video_path, output_folder, resize_shape=OUT_HEIGHT_WIDTH, shortest_side_size=480
    )
