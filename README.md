## **HD-Set: Human Disability 2D Image Dataset**

📌 **Dataset Name**: **HD-Set**  
📌 **Total Images**: **3,877**  
📌 **Classes**: **7 (Six disability types + Normal category for comparison)**  

### **📖 Overview**
The **HD-Set** dataset consists of **3,877** high-quality **2D images** of individuals with different disabilities. This dataset serves as a valuable resource for **computer vision, medical AI research, and accessibility studies**. It includes six distinct disability classes alongside a **normal** category to facilitate comparison.  

---

### **📂 Dataset Composition**
The dataset contains **the following classes**:

| Class Name         | Number of Images |
|--------------------|----------------|
| Down Syndrome     | 1,123          |
| Blind            | 543            |
| Dwarf            | 448            |
| Prosthetic Legs  | 872            |
| Prosthetic Arms  | 334            |
| Cerebral Palsy   | 557            |
| **Normal**       | 1,000          |
| **Total**        | **3,877**      |

All images were collected from **royalty-free datasets** to ensure compliance with **data sharing ethics**.

---

### **🛠️ Usage & Applications**
HD-Set is useful for:
- **Machine Learning & Deep Learning**: Training and evaluating models for **disability detection**.
- **Medical Research**: Assisting in studies related to **disability identification**.
- **Accessibility Projects**: Improving AI-powered accessibility tools.
- **Human-Centered AI**: Enhancing applications in **healthcare and social inclusion**.

---

### **⚙️ Project Structure and Scripts**
The repository includes the following key scripts:

- `data_prepare.py`: Resizes images while preserving aspect ratio and creates 10-fold cross-validation splits.
- `test.py`: Runs inference using a trained detection model (TensorFlow or Detectron2 RetinaNet) on a single image.
- `train.py`: Loads and trains various models including:
  - TensorFlow-based: `CenterNet`, `EfficientDet`, `Faster R-CNN`, `SSD`
  - Detectron2-based: `RetinaNet`
- `crop_image.py`: Applies **center cropping** to all images in a given folder and saves the cropped outputs.

Each script includes command-line arguments for flexibility and integration.

---

### **📦 Requirements**
To install required packages, use the following:

```bash
pip install -r requirements.txt
```

#### `requirements.txt` should include:
```
tensorflow>=2.10
Pillow
scikit-learn
opencv-python
matplotlib
# TensorFlow Object Detection API dependencies
lxml
Cython
contextlib2
pycocotools
# Detectron2 (install via GitHub)
# pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

---

### **📥 Download**
To access the dataset and code, clone this repository:
```bash
git clone https://github.com/yourusername/HD-Set.git
```

---

### **📜 Citation**
If you use this dataset in your research, please cite:
```bibtex
@dataset{HD-Set2025,
  author = {Your Name},
  title = {HD-Set: Human Disability 2D Image Dataset},
  year = {2025},
  publisher = {GitHub Repository},
  url = {https://github.com/yourusername/HD-Set}
}
```

---

### **📄 License**
This dataset is available under the **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)** license. You are free to use, modify, and distribute it **with proper attribution**.

---

### **🙌 Contribution**
If you would like to improve the dataset or scripts (e.g., adding annotations or new preprocessing functions), feel free to submit a **pull request** or open an **issue**.
