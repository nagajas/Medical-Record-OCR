# OCR on Patient Records

### Objective

The objective is to propose a comprehensive workflow to perform Optical Character Recognition (OCR) on medical records written by hand by medical practitioners so as to aid the database management in hospitals, in one of the applications. Further, it can help general public understand the record more comprehensively when comined with Natural Language Processing (NLP). This procedure

### Parameters

Takes a patient record in the form of image (.jpg) as input and performs OCR to return information in form of text, related to bio data and medical data for further processing by users.

### Workflow

Brief introduction to the workflow is provided here, please refer to report for detailed explanation. The workflow includes two major procedures.

1. **Preprocessing:** This step involves loading of image data and performing cleaning on images, which includes edge detection, orientation detection, page segmentation, etc.
2. **OCR:** Here, we take the cleaned image and perform labelling using keras, tensorflow, etc., and then build a new model on top of existing OCR model, tesseract.
