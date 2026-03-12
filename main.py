import Camera_helper.py


while True:
    print("Waiting for button...")

    Camera_helper.wait_for_press()

    print("Recording workout")

    images = Camera_helper.capture_images()

    print("Sending to cloud")

    suggestion = Camera_helper.send_to_cloud(images)

    image_capture_send_eval.Camera_helper.delete_images(images)

    print("Feedback:", suggestion)

    image_capture_send_eval.Camera_helper.speak_feedback(suggestion)