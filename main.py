import Camera_helper.py


while True:
    print("Waiting for button...")

    Camera_helper.wait_for_press()

    print("Recording workout")

    images = Camera_helper.capture_images()

    print("Sending to cloud")

    suggestion = Camera_helper.send_to_cloud(images)

    Camera_helper.delete_images(images)

    print("Feedback:", suggestion)

    Camera_helper.speak_feedback(suggestion)