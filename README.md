# Currency Converter

An application that takes in images of menus, price tags, and different currencies and outputs conversions for the desired currency.

If you have any questions, feel free to open an issue on [Github](https://github.com/organization-x/omni/issues).

## Development Process

### Topic Selection

We began by collecting project ideas from the group, then we assessed the viability from a technological standpoint and potential use case. After initially settling on facial recognition, we switched to a currency converter due to the technical challenges associated with facial recognition training and a limited use case.

### Gathering Data

We then set out to gather data on banknotes, price tags, and menus from around the world that could use to train an AI model. We collected images in Roboflow and utilized them to train Yolov5 and Yolov7 models in Google Colab and Runpod.

### Conversion

After identifying the image, it is passed to EasyOCR to lift the bill's denomination off the not. We also wrote a function to take the EXIF GPS data from the photo to identify the country in which the person took the photo and then determine the currency used there.

### Final Product

The final product accepts an image of a price tag, menu, or bill. It identifies the images and locates the price\(s\) on them. It combines the price with the GPS location to attain conversion rates. Finally, it projects the price of the image in a more familiar currency.
