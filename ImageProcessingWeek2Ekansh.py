import cv2
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the images
# Replace 'image1.jpg' and 'image2.jpg' with your actual file paths
img1 = cv2.imread('image1.png')
img2 = cv2.imread('image2.png')

# Convert from BGR to RGB for correct Matplotlib visualization
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

def process_image(img_rgb, title_suffix=""):
    # Step 2: Grayscale Conversion
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Step 3: Image Resizing (Resizing to 300x300 pixels)
    interpolation=int(input("pick 1 for downscaling , 2 for upscaling to (300,300) pixels"))
    if (interpolation==1):
        img_resized = cv2.resize(img_gray, (300, 300), interpolation=cv2.INTER_AREA)
    elif(interpolation==2):
        img_resized = cv2.resize(img_gray, (300, 300), interpolation=cv2.INTER_CUBIC)
    
    # Step 4: Edge Detection (Canny) for 50 ,150 min max values for 1:3 rule
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0) # (5,5) blurring done on kernal if u put (0,0) then thats automatically chosen , 0 standard deviation , means it decides on its own the intensity required
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # Step 5: Noise Addition (Gaussian Noise)
    row, col = img_resized.shape #represented as 2d array thats why row , col can be extracted  
    mean = 0
    sigma = 25  # Standard deviation for noise intensity
    gauss = np.random.normal(mean, sigma, (row, col)) #generates a matrix of the same size as our image containing random numbers based on a normal distribution curve
    noisy_image = np.clip(img_resized + gauss, 0, 255).astype(np.uint8)# adds gauss image to resized image , while astype(np.uint8) is just making sure it stays unsigned integer 8 bits ,
    
    # Step 6: Noise Removal (Gaussian Blur Filter)
    denoised_image = cv2.GaussianBlur(noisy_image, (5, 5), 0) #(5,5) filter used to create a feature map AND UNFBLUR IT 
    #to make it more denoised , either increase filter size or use cv2.medianBlur(image,kernel)
   # denoised_image = cv2.GaussianBlur(denoised_image, (5, 5), 0) , optional repeated denoising

    # Visualizing the Results
    plt.figure(figsize=(15, 10)) #creates new empty window 15x10 inches
    
    # Plot Original (RGB)
    plt.subplot(2, 4, 1) # r x c , position=1 means top left column its placed in 
    plt.imshow(img_rgb) #in rgb format
    plt.title(f'Original {title_suffix}') #title suffix depends on image passed
    plt.axis('off')
    
    # Plot Grayscale
    plt.subplot(2, 4, 2) # psn2 means top middle
    plt.imshow(img_gray, cmap='gray') #without this matplotlib doesnt make images actually gray
    plt.title('Grayscale')
    plt.axis('off')
    
    # Plot Histogram
    plt.subplot(2, 4, 3) # 3 means right top
    plt.hist(img_gray.ravel(), bins=256, range=[0,256], color='gray') #variable.ravel()makes it a 1D array since histogram needs that, with 256 divisions as gray scale has 256 shades , with x axis max
    plt.title('Histogram')
    
    # Plot Resized
    plt.subplot(2, 4, 4)# 4 means middle left
    plt.imshow(img_resized, cmap='gray')
    plt.title('Resized (300x300)')
    plt.axis('off')
    
    # Plot Edges
    plt.subplot(2, 4, 5)# 5 means middle middle
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')
    
    # Plot Noisy Image
    plt.subplot(2, 4, 6)# 6 means bottom left
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Added Noise')
    plt.axis('off')
    
    # Plot Denoised Image
    plt.subplot(2, 4, 7) # means bottom middle
    plt.imshow(denoised_image, cmap='gray')
    plt.title('Gaussian Filter (Denoised)')
    plt.axis('off')
    
    plt.tight_layout() #readjustment
    plt.savefig(f'results_{title_suffix.lower().replace(" ", "")}.png', dpi=300) #dpi = dots per inch , our screen is 15x 10 so 4500x3000 pixel photo shud be saved
    plt.show() #finally done

# Run the pipeline for both images
process_image(img1_rgb, title_suffix="Image 1")
process_image(img2_rgb, title_suffix="Image 2")