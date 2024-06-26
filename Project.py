#----- DIP Final Project Sec: A -----#
#------- i210465 Hassan Raza --------#
#------- i212483 Moiez Asif ---------#
#------- i210375 Aasir Farrukh ------#

# Imported Libraries
import numpy as np  # For numerical operations
import cv2  # For image processing
import os  # For operating system dependent functionality
from tensorflow.keras.models import Model  # For creating the model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate  # For creating the model layers
from tensorflow.keras.optimizers import Adam  # For the optimizer
from tensorflow.keras.losses import binary_crossentropy  # For the loss function
from tensorflow.keras.metrics import MeanIoU  # For the metric
from tensorflow.keras.models import load_model  # For loading a saved model
import tkinter as tk  # For creating the GUI
from tkinter import filedialog, Scrollbar  # For the file dialog and the scrollbar in the GUI
from PIL import ImageTk, Image  # For handling images in the GUI
import matplotlib.pyplot as plt  # For plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # For displaying matplotlib plots in the GUI
from matplotlib.figure import Figure  # For creating a new figure
from tkinter import ttk  # For themed widgets

# Define function for U-Net model for segmentation
def unet(inputSize=(256, 256, 1)):
    # Define the input layer
    inputs = Input(inputSize)  
    # Encoder
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)
    # Decoder
    up1 = concatenate([UpSampling2D(size =(2, 2))(conv2), conv1], axis = -1)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same')(up1)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv3)
    conv4 = Conv2D(1, 1, activation = 'sigmoid')(conv3)
    model = Model(inputs = inputs, outputs = conv4)
    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = binary_crossentropy, metrics = [MeanIoU(num_classes = 2)])  
      
    return model

# Function to load image and mask data
def loadData(imagePath, maskPath):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    # Error handling
    if image is None:
        raise FileNotFoundError(f"Could not read image at {imagePath}")
       
    # Apply Gaussian filter to smooth the image
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Resize image
    image = cv2.resize(image, (256, 256))
    # Normalize only if necessary
    if np.max(image) > 1:
        image = image / 255.0
    
    mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask at {maskPath}")
    
    mask = cv2.resize(mask, (256, 256))
    mask = np.expand_dims(mask, axis=-1)
    # Normalize only if necessary
    if np.max(mask) > 1:
        mask = mask / 255.0

    return image, mask

# Function to train the segmentation model
def trainModel(numSamples):
    model = unet() 
    # Load data
    images, masks = [], []
    for i in range(1, numSamples):
        imagePath = f'RIGA 200 Images/{i}.png'
        maskPath = f'RIGA 200 Images/{i}.tif'
        image, mask = loadData(imagePath, maskPath)
        images.append(image)
        masks.append(mask)
    
    images = np.array(images)
    masks = np.array(masks)
    
    # Train the model
    model.fit(images, masks, batch_size=8, epochs=10, validation_split=0.2)

    return model

# Segment optic disc
def segmentOpticDisc(imagePath, model):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)  # Add an extra dimension for the channel
    # Normalize only if necessary
    if np.max(image) > 1:
        image = image / 255.0
    
    image = image.astype(np.float32)  # Convert to float32
    # Predict segmentation mask
    mask = model.predict(image)
    mask = np.squeeze(mask)    
    # Invert the mask
    mask = cv2.bitwise_not(mask)

    return mask

# Segment optic cup
def segmentOpticCup(opticDiscMask):
    # Convert the optic disc mask to uint8
    discMask = opticDiscMask.astype(np.uint8) 
    # Find the unique values in the disc mask
    uniqueValues = np.sort(np.unique(discMask))
    
    # The darkest color (the cup) will have the lowest pixel value
    cupValue = uniqueValues[0]
    
    # The second darkest color (the disc) will have the next lowest pixel value
    discValue = uniqueValues[1]
    
    # Create a binary mask for the cup
    cupMask = np.where(discMask == cupValue, 255, 0).astype(np.uint8)
    
    # Create a binary mask for the disc
    discMask = np.where(discMask == discValue, 255, 0).astype(np.uint8)
    
    # Define the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Erode the disc mask
    discMask = cv2.dilate(discMask, kernel, iterations = 3)
    discMask = cv2.erode(discMask, kernel, iterations = 3)

    # Apply morphological dilation to the cup mask
    cupMask = cv2.dilate(cupMask, kernel, iterations = 2)

    return discMask, cupMask

# Calculate cup-to-disc ratio
def calculateCdr(opticDiscMask, opticCupMask):
    #cdr = np.sum(optic_cup_mask) / (np.sum(optic_disc_mask) + np.sum(optic_cup_mask))
    cdr = np.sum(opticDiscMask) / (np.sum(opticCupMask))
    return cdr

# Create a global variable to hold the canvas widget
canvas = None

# Function to select an image
def selectImage():
    global canvas  # Declare the canvas variable as global
    # Load trained model
    trainedModel = load_model(os.path.join('models','segment.h5'))
    imagePath = filedialog.askopenfilename()
    
    # Load and segment the optic disc
    opticDiscMask = segmentOpticDisc(imagePath, trainedModel)

    # Segment the optic cup
    disc, cup = segmentOpticCup(opticDiscMask)
    # Calculate CDR
    cdr = calculateCdr(disc, cup)
    # Display the CDR
    cdrLabel.config(text=f"Cup-to-disc ratio: {cdr}")

    # Load the original image and ground truth mask
    originalImage = cv2.imread(imagePath)
    #convert the image path to .tif when calling ground truth mask
    groundTruthMask = cv2.imread(imagePath.replace('.png', '.tif'), cv2.IMREAD_GRAYSCALE)  # replace with the correct path
    #ground_truth_mask = cv2.imread('RIGA 200 Images/92.tif', cv2.IMREAD_GRAYSCALE)

    # Create a new figure and add subplots
    fig = Figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 5, 1)
    ax2 = fig.add_subplot(1, 5, 2)
    ax3 = fig.add_subplot(1, 5, 3)
    ax4 = fig.add_subplot(1, 5, 4)
    ax5 = fig.add_subplot(1, 5, 5)

    # Plot the original image
    ax1.imshow(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')

    # Plot the ground truth mask
    ax2.imshow(groundTruthMask, cmap='gray')
    ax2.set_title('Ground Truth Mask')

    # Plot the predicted optic disc mask without inverting
    ax3.imshow(opticDiscMask, cmap='gray')  # No need to multiply by 255
    ax3.set_title('Predicted Optic Disc Mask')

    # Plot the optic disc mask
    ax4.imshow(disc * 255, cmap='gray')  # Multiply by 255 to display correctly
    ax4.set_title('Optic Disc Mask')

    # Plot the optic cup mask
    ax5.imshow(cup * 255, cmap='gray')  # Multiply by 255 to display correctly
    ax5.set_title('Optic Cup Mask')

    # Remove the old canvas from the root window
    if canvas is not None:
        canvas.get_tk_widget().pack_forget()

    # Create a new canvas and add it to the root window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Function to train the model
def train():
    numSamples = 10
    model = trainModel(numSamples)
    model.save(os.path.join('models','segment.h5'))
    # show message model trained
    tk.messagebox.showinfo("Model Trained", "Model has been trained successfully")

# Function to exit the program
def exit_program():
    root.quit()

# GUI Styling
# Create the main window
root = tk.Tk()
root.title("Optic Disc and Cup Segmentation")  # Set the title of the window
root.geometry("1200x600")  # Set the initial size of the window
# Set the background color of the window
root.configure(bg = '#000000')

# Button to select an image
button = ttk.Button(root, text = "Select Image", command = selectImage, style = 'TButton')
button.pack(side = tk.TOP, pady = 10)  # Add some padding around the button

# Button to train the model
trainButton = ttk.Button(root, text = "Train Model", command = train, style = 'TButton')
trainButton.pack(side = tk.TOP, pady = 10)  # Add some padding around the button

# Button to exit the program
exitButton = ttk.Button(root, text = "Exit", command = root.quit, style = 'TButton')
exitButton.pack(side = tk.BOTTOM, pady = 10)  # Add some padding around the button

# Labels to display the Headings and members
label1 = tk.Label(root, text = "Glaucoma Detection", font = ("Arial", 24), bg = '#000000', fg = '#ffffff')
label1.pack(side = tk.TOP, pady = 20)
# Member 1
label2 = tk.Label(root, text = "Hassan Raza i210465", font = ("Arial", 16), bg = '#000000', fg = '#ffffff', anchor = 'center')
label2.place(relx = 0.5, rely = 0.75, anchor = 'center')
# Member 2
label3 = tk.Label(root, text = "Moiez Asif i212483", font = ("Arial", 16), bg = '#000000', fg = '#ffffff', anchor = 'center')
label3.place(relx = 0.5, rely = 0.8, anchor = 'center')
# Member 3
label4 = tk.Label(root, text = "Aasir Farrukh i210375", font = ("Arial", 16), bg = '#000000', fg = '#ffffff', anchor = 'center')
label4.place(relx = 0.5, rely = 0.85, anchor = 'center')
# Label to display the CDR
cdrLabel = tk.Label(root, text = "", bg = '#000000', fg = '#ffffff')
cdrLabel.pack(side = tk.TOP, pady = 10)

# Create a style object
style = ttk.Style()
# Configure the style for the buttons
style.configure('TButton', foreground = 'black', background = '#6ac6d4', font = ('Arial', 12), padx = 10, pady = 5)

# Run the GUI
root.mainloop()