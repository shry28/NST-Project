<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links-->
<div align="center">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

</div>

<!-- PROJECT LOGO -->
<br />
<!-- UPDATE -->
<div align="center">
  <a href="https://github.com/cgs-iitkgp/NST-Project">
     <img width="500" alt="Neural Style Transfer Example" src="https://raw.githubusercontent.com/shry28/NST-Project/main/generated_img/generated_blener_vg.png">
  </a>

  <h3 align="center">NST-Project</h3>

  <p align="center">
  <!-- UPDATE -->
    <i>Turn any photo into an artistic masterpiece using Neural Style Transfer</i>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
<summary>Table of Contents</summary>

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)
  - [Maintainer(s)](#maintainers)
  - [creators(s)](#creators)
- [Additional documentation](#additional-documentation)

</details>


<!-- ABOUT THE PROJECT -->
## About The Project
<!-- UPDATE -->
<div align="center">
  <a href="https://github.com/cgs-iitkgp/NST-Project">
    <img width="80%" src="https://raw.githubusercontent.com/shry28/NST-Project/main/examples/result_vangogh.jpg">
  </a>
</div>

Neural Style Transfer (NST) is a deep learning technique that merges the content of one image with the style of another. This project uses PyTorch to implement NST, allowing you to generate stylized images with pre-trained VGG-19 features. You can experiment with different styles, content images, and tuning parameters to get unique results.

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

To set up a local instance of the application, follow the steps below.

### Prerequisites
The following dependencies are required to be installed for the project to function properly:
<!-- UPDATE -->
* Python 3.8+
* PyTorch
* torchvision
* Pillow

  ```sh
  pip install -r requirements.txt
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Installation

_Now that the environment has been set up and configured to properly compile and run the project, the next step is to install and configure the project locally on your system._
<!-- UPDATE -->
1. Clone the repository
   ```sh
   git clone https://github.com/shry28/NST-Project.git
   cd NST-Project
   ```
2. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```
3. Execute the script
   ```sh
   python nst.py --content path/to/content.jpg --style path/to/style.jpg --output output.jpg
   ```

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage
<!-- UPDATE -->
Once installed, you can run the script from the command line to apply style transfer.

<div align="center"> <img src="https://raw.githubusercontent.com/shry28/NST-Project/main/content img/pool.jpg" width="30%" alt="Content Image"> <img src="https://raw.githubusercontent.com/shry28/NST-Project/main/style type/oil.jpg" width="30%" alt="Style Image"> <img src="https://raw.githubusercontent.com/shry28/NST-Project/main/generated_img/generated_pool_oil.png" width="30%" alt="Styled Output"> </div>

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact

### Maintainer(s)

The currently active maintainer(s) of this project.

<!-- UPDATE -->
- [Shaurya Choudhary](https://github.com/shry28)

### Creator(s)

Honoring the original creator(s) and ideator(s) of this project.

<!-- UPDATE -->
- [Shaurya Choudhary](https://github.com/shry28)

<p align="right">(<a href="#top">back to top</a>)</p>

## Additional documentation

  - [License](/LICENSE)
 

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

[contributors-shield]: https://img.shields.io/github/contributors/shry28/NST-Project.svg?style=for-the-badge
[contributors-url]: https://github.com/shry28/NST-Project/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shry28/NST-Project.svg?style=for-the-badge
[forks-url]: https://github.com/shry28/NST-Project/network/members
[stars-shield]: https://img.shields.io/github/stars/shry28/NST-Project.svg?style=for-the-badge
[stars-url]: https://github.com/shry28/NST-Project/stargazers
[issues-shield]: https://img.shields.io/github/issues/shry28/NST-Project.svg?style=for-the-badge
[issues-url]: https://github.com/shry28/NST-Project/issues
[license-shield]: https://img.shields.io/github/license/shry28/NST-Project.svg?style=for-the-badge
[license-url]: https://github.com/shry28/NST-Project/blob/main/LICENSE
