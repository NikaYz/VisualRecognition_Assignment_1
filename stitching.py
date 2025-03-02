import cv2
import numpy as np
import matplotlib.pyplot as plt


# Stitch the images given in the list
def stitch_images(image_list):
    # to handle case when there are no only one image in the list
    if len(image_list) < 2:
        raise ValueError("Warning: At least two images are required for stitching.")

    # Take the first image as a reference or start point
    base_image = cv2.imread(image_list[0])
    
    # Traverse through other images in the list
    for i in range(1, len(image_list)):
        next_image = cv2.imread(image_list[i])

        # Converting images to grayscale
        gray1 = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

        # Use SIFT to detect keypoints and descriptors for the imagaes which are important for images
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        # Use FLANN for feature matching
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Apply Lowe’s ratio test : It's done to get matches which are closer in distance in both figures, Not like one pont is in total left and other point in different image is in total right
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        
        # if there are less than 4 matches then we don't do stitching
        if len(good_matches) < 4:
            print(f"Not enough good matches for {image_list[i]} - Skipping")
            continue

        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Get dimensions of images
        height1, width1 = base_image.shape[:2]
        height2, width2 = next_image.shape[:2]

        # Transform corners of base image to new perspective
        corners1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
        transformed_corners1 = cv2.perspectiveTransform(corners1, H)

        # Get the dimensions of the final stitched image
        min_x = min(0, np.min(transformed_corners1[:, 0, 0]))
        min_y = min(0, np.min(transformed_corners1[:, 0, 1]))
        max_x = max(width2, np.max(transformed_corners1[:, 0, 0]))
        max_y = max(height2, np.max(transformed_corners1[:, 0, 1]))

        # Translation matrix to shift stitched image into visible range
        translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

        # Compute final stitched image size
        stitched_width = int(max_x - min_x)
        stitched_height = int(max_y - min_y)

        # Warp base image
        warped_image = cv2.warpPerspective(base_image, translation_matrix @ H, (stitched_width, stitched_height))

        # Overlay next image into the stitched canvas
        x_offset = -int(min_x)
        y_offset = -int(min_y)
        warped_image[y_offset:y_offset+height2, x_offset:x_offset+width2] = next_image

        # Update base_image with the newly stitched image
        base_image = warped_image

    return base_image




# image files path
image_files = ["images/stitching/clicked/1.jpeg", "images/stitching/clicked/2.jpeg", "images/stitching/clicked/3.jpeg", "images/stitching/clicked/4.jpeg", "images/stitching/clicked/5.jpeg", "images/stitching/clicked/6.jpeg"]  # Add more images as needed
image_files2 = ["images/stitching/clicked2/a.jpeg", "images/stitching/clicked2/b.jpeg", "images/stitching/clicked2/c.jpeg"]  # Add more images as needed
image_files3 = ["images/stitching/online/1.jpg", "images/stitching/online/2.jpg"]  # Add more images as needed

# Perform stitching on different set of images
stitched_result = stitch_images(image_files)
stitched_result2 = stitch_images(image_files2)
stitched_result3 = stitch_images(image_files3)

# Display images

plt.figure(figsize=(8, 4))
plt.imshow(cv2.cvtColor(stitched_result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Stitched Panorama 1")
plt.show()

plt.figure(figsize=(8, 4))
plt.imshow(cv2.cvtColor(stitched_result2, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Stitched Panorama 2")
plt.show()

plt.figure(figsize=(8, 4))
plt.imshow(cv2.cvtColor(stitched_result3, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Stitched Panorama 3")
plt.show()


# Save images
plt.imsave("output/2/stitched_output_clicked.jpg", stitched_result) 
plt.imsave("output/2/stitched_output_clicked2.jpg", stitched_result2) 
plt.imsave("output/2/stitched_output_online.jpg", stitched_result3) 
