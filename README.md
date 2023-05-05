# House Price Prediction via Computer Vision

This project aims to predict house prices from scraped exterior frontal images using a Convolutional Neural Network (CNN) and various preprocessing techniques. The dataset contains over 20,000 images and numerical data for single-family homes in Southern California. The model, trained on this dataset, can predict house prices with an average error of 4.3%.

## Summary

The [Blog Post](https://nycdatascience.com/blog/student-works/predicting-house-prices-from-scraped-exterior-front-images/) describes the process of predicting house prices using a combination of numerical data and images. . I created a tool to collect data from Weichert's website for Southern California listings. The dataset consists of exterior frontal images (about 15000) and numerical information (7 columns) such as the number of bathrooms and bedrooms, square footage, address, and price. I made this dataset publicly available on [Kaggle](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal). The model, a convolutional neural network (CNN), was trained and tested for various classification tasks such as binary and categorical classification, and finally predicting house prices.

## Prerequisites

Ensure that you have the following libraries installed:

    selenium
    pandas
    scikit-learn
    numpy
    glob
    cv2
    Keras

 
 Currently the majority of the machine learning modeling done cannot capture what it means to see a place that you know you’d like to call home, and the model suffers for it. these unquantifiable features that could tune the accuracy of a model by a few percent. For an industry that is at around $30T, and knowing that half the buyers decide by the look, by extracting the images of properties along with other data, i will try to quantify the unquantifiable.
 
## Model

The model uses a Convolutional Neural Network (CNN) for binary and categorical classification. It is trained to differentiate various categories such as house or car, house or car or neither, and kitchen, living room, bedroom, bathroom, or front yard. The model achieves an overall accuracy of 75% in categorical classification and 54% in classifying low, medium, and high-priced houses.

 
  
 
	6+1 method of estimating housing market (model that I created)
	
	Web Scraping by Selenium
		
	house or car (binary)
	
	house or car or neither (categorical)
	
	kitchen or living room or bedroom or bathroom or frontyard (categorical)
	
	low priced or medium priced or high priced
	
		economic reasoning, zillow report, and previous research
		
	price prediction by images
 
	NEXT: object detection and measuring...
	
	LIMITATIONS: computation power, more images (interior and exterior), tuning ConvNet 
 
 

 ## Conclusion

The project demonstrates the potential of using machine learning, specifically convolutional neural networks, to predict house prices based on exterior frontal images and numerical data. The method currently has limitations in terms of computation power, the need for more images (both interior and exterior), and tuning of the CNN model. However, with further improvements, it has the potential to become a valuable tool for accurately predicting house prices, saving both time and money in the real estate industry.