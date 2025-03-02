import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Load image

#image_path = 'images/coin/coins1.jpeg'
image_path = 'images/coin/coins2.jpeg'
image = cv2.imread(image_path)

# To use filename while saving image
filename = os.path.splitext(os.path.basename(image_path))[0]

# Convert to grayscale and apply Gaussian blur (to remove noise)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)

# Compute histogram for grayscale pixel distribution
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Apply Edge Detection Techniques
edges_Canny = cv2.Canny(blur, 50, 150)
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobelxy = cv2.magnitude(sobelx, sobely)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Convert Sobel & Laplacian to uint8 for contour detection
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.convertScaleAbs(sobelxy)
laplacian = cv2.convertScaleAbs(laplacian)

# Threshold the images for better contour detection as normally it leds to detection of images of not coin
_, sobelx_bin = cv2.threshold(sobelx, 150, 255, cv2.THRESH_BINARY)
_, sobely_bin = cv2.threshold(sobely, 150, 255, cv2.THRESH_BINARY)
_, sobelxy_bin = cv2.threshold(sobelxy, 150, 255, cv2.THRESH_BINARY)
_, laplacian_bin = cv2.threshold(laplacian, 60, 255, cv2.THRESH_BINARY)

# Apply thresholding and invert to get coins image in white
threshold_value = 70
_, binary_thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
binary_thresh = cv2.bitwise_not(binary_thresh)

# Figure 1 : Original Image and histogram
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))

# To show original image
axes1[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes1[0].set_title("Original Image", fontsize=12)
axes1[0].axis("off")

# To show histogram of distribution of pixels of the image
axes1[1].plot(hist, color='black')
axes1[1].set_title("Pixel Intensity Distribution", fontsize=12)
axes1[1].set_xlabel("Pixel Value")
axes1[1].set_ylabel("Frequency")

# To view and save the figure 1
plt.tight_layout()
plt.savefig(f"output/coins/{filename}/OriginalImage_FrequencyDistribution.png", dpi=300, bbox_inches='tight')
plt.show()


# Figure-2 : Edge Detection
titles = ["Canny Edge", "Sobel X", "Sobel Y", "Sobel Magnitude", "Laplacian", "Inverted Thresholding"]
edges = [edges_Canny, sobelx_bin, sobely_bin, sobelxy_bin, laplacian_bin, binary_thresh]

fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))

for ax, edge, title in zip(axes2.flat, edges, titles):
    ax.imshow(edge, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

# To show and save figure 2
plt.tight_layout()
plt.savefig(f"output/coins/{filename}/EdgeDetection.png", dpi=300, bbox_inches='tight')
plt.show()


# Figure 3 : Count number of coins in edge detection
fig3, axes3 = plt.subplots(2, 3, figsize=(12, 8))

for ax, edge, title in zip(axes3.flat, edges, titles):
    # To dilate edges to close small gaps
    dilated = cv2.dilate(edge.astype(np.uint8), None, iterations=2)

    # To find contours
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # To draw contours on a copy of the original image
    coins_detected = image.copy()
    cv2.drawContours(coins_detected, contours, -1, (0, 255, 0), 2)

    ax.imshow(cv2.cvtColor(coins_detected, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{title}\nCoins: {len(contours)}", fontsize=10)
    ax.axis("off")

# to show and save image of count of coins using edge detection
plt.tight_layout()
plt.savefig(f"output/coins/{filename}/CountCoinsEdgeDetection.png", dpi=300, bbox_inches='tight')
plt.show()

# Part 2 : Using segmentation approach

# This was done again as this reduce more noise and give proper result 
blur = cv2.GaussianBlur(gray, (11,11), 0)

_, binary_thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

# Contour Detection Methos
contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
num_coins = len(contours)
segmented_thresh = image.copy()
cv2.drawContours(segmented_thresh, contours, -1, (0, 255, 0), 2)
cv2.putText(segmented_thresh, f"Coins: {num_coins}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# segmented outputs for each detected coin
segmented_coins = []
for i, contour in enumerate(contours):
    mask = np.zeros_like(binary_thresh)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    segmented_coin = cv2.bitwise_and(image, image, mask=mask)
    segmented_coins.append(segmented_coin)

# Region Growing approach
def region_growing(image, seed, threshold=10):
    height, width = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)
    
    stack = [seed]
    seed_value = image[seed]
    
    while stack:
        y, x = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = True
        if abs(int(image[y, x]) - int(seed_value)) < threshold:
            segmented[y, x] = 255  # Mark as part of region
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                    stack.append((ny, nx))
    
    return segmented

seed_point = (100, 100)

segmented_region_growing = region_growing(binary_thresh, seed_point)
segmented_region_growing = cv2.bitwise_not(segmented_region_growing)
cv2.putText(segmented_region_growing, f"Coins: {num_coins}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Used to plot the images of segmentation approach
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), ax[0, 0].set_title("Original Image")
ax[0, 1].imshow(binary_thresh, cmap="gray"), ax[0, 1].set_title(f"Thresholding + Contours\nCoins: {num_coins}")
ax[1, 0].imshow(segmented_region_growing, cmap="gray"), ax[1, 0].set_title(f"Region Growing\nCoins: {num_coins}")

ax[1, 1].imshow(cv2.cvtColor(segmented_thresh, cv2.COLOR_BGR2RGB)), ax[1, 1].set_title(f"Contours Detected\nCoins: {num_coins}")

# to show and save image for the segmentation of coins
plt.tight_layout()
plt.savefig(f"output/coins/{filename}/Segmentation.png", dpi=300, bbox_inches='tight')
plt.show()


# Used to show segmented coins done by segmentation show in green box
fig2, ax2 = plt.subplots(1, len(segmented_coins), figsize=(10, 4))

if len(segmented_coins) == 1:
    ax2 = [ax2]

for i, segmented_coin in enumerate(segmented_coins):
    gray = cv2.cvtColor(segmented_coin, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlighted_coin = segmented_coin.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(highlighted_coin, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    ax2[i].imshow(cv2.cvtColor(highlighted_coin, cv2.COLOR_BGR2RGB))
    ax2[i].set_title(f"Segmented Coin {i+1}")
    ax2[i].axis("off")

# to show and save image for Segmented coins
plt.tight_layout()
plt.savefig(f"output/coins/{filename}/SegmentedCoins.png", dpi=300, bbox_inches='tight')
plt.show()


