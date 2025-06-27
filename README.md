# Neural Style Transfer – PyTorch Implementation

This project implements Neural Style Transfer (NST) using PyTorch, allowing you to blend the content of one image with the artistic style of another to generate a new, stylized image.

Inspired by the paper _"A Neural Algorithm of Artistic Style"_ by Gatys et al., this code uses a pre-trained VGG19 network to compute content and style representations and optimize a new image to match them.

## 🖼️ Example Output

The generated image combines the structure of the content image with the texture and color patterns of the style image.

![Output Image](generated_img.jpg)

## 🧠 Features

- Feature extraction using VGG19 from `torchvision.models`
- Layer-wise content and style loss calculation
- Balancing style and content via adjustable weights (`alpha`, `beta`)
- Image optimization via backpropagation and gradient descent
- Includes a project report with explanation and results

## 📁 Files Included

- `RnDTask_NST.py` – Main NST implementation script
- `content.jpg` – Content image
- `style.jpg` – Style image
- `generated_img.jpg` – Final stylized output
- `NST_Project_Report.pdf` or Google Docs link – Report with explanation and results

## 🛠️ Requirements

Install dependencies with:
pip install torch torchvision pillow matplotlib

##▶️ How to Use
Replace the image paths inside RnDTask_NST.py with your own:

python
content_img = load_image("content.jpg")
style_img = load_image("style.jpg")
Run the script:

bash
python RnDTask_NST.py
The result will be saved as generated_img.jpg.

##📄 Report
A detailed project report is included in the repository.
📎 [View the report on Google Docs](https://docs.google.com/document/d/19knktGb8CzOxJmz6Laj1xA6S1A__GzoIEnU0gV6-GjE/edit?tab=t.0)

##📚 References
Gatys, Leon A., et al. “A Neural Algorithm of Artistic Style.” 2015.
PyTorch NST tutorial

```bash
