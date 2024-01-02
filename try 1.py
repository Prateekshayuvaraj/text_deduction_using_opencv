import cv2    
#importing lib

class TextDeductor:
    
    def __init__(self, image_path):
        self.image_path = image_path

    def preprocess_image(self):
        # Read image from which text needs to be extracted
        self.img = cv2.imread(self.image_path)                

        # Convert the image to grayscale
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Perform OTSU threshold
        _, self.thresh1 = cv2.threshold(self.gray, 0, 200, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # Specify structure shape and kernel size it is adjused to get the perfect value
        self.rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    def find_text_contours(self):
        # Apply dilation on the threshold image (enancing)
        dilation = cv2.dilate(self.thresh1, self.rect_kernel, iterations=1)

        # Find contours 
        self.contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    def remove_small_boxes(self, min_area):
        # Remove small contours based on the specified minimum area
        self.contours = [cnt for cnt in self.contours if cv2.contourArea(cnt) > min_area]

    def deduct_text(self):
        # Creating a copy of the image
        im2 = self.img.copy()

        # Loop through the identified contours
        for cnt in self.contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw a rectangle on the copied image
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Cropping the text block for further processing (if needed)
            cropped = im2[y:y + h, x:x + w]

            # Here, you can add additional processing or save the cropped region as needed

        # Display the result
        cv2.imshow ("Text Detection", im2)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to your image file
    
    image_path = 'image 3.png'

    # Create an instance of the TextDeductor class
    text_deductor = TextDeductor(image_path)

    # Preprocess the image
    text_deductor .preprocess_image()

    # Find text contours
    text_deductor .find_text_contours()
    
    text_deductor.remove_small_boxes(min_area=5)

    # Extract and display text
    text_deductor .deduct_text()
