<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Python][Python.org]][Python-url]



<h1 align="center">Exposure Fusion</h1>

  <p align="center">
    An implementation of the Exposure Fusion paper through a fast, lightweight Python class, inspired
    by <code>pytorch</code> modules.
    <br />
    <br />
    <a href="https://github.com/dario-loi/exposure-fusion/issues">Report Bug</a>
    ·
    <a href="https://github.com/dario-loi/exposure-fusion/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Sample Image][product-screenshot]](https://example.com)

The project was created as classwork for the course *Computer Graphics* at Sapienza University
of Rome, it is licensed under the MIT license and is free to use for any purpose.

It improves on the Stanford paper [Exposure Fusion](https://web.stanford.edu/class/cs231m/project-1/exposure-fusion.pdf), by adding an alignment pipeline that can improve outputs when the input sequence's photos are offset from eachother.

The repository contains both a class `exposure_fusion.py` and a TKinter GUI `gui.py` that 
can use the class in a user friendly way`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To use the class it is enough to copy the `exposure_fusion.py` file into your project and import it.

```python

from exposure_fusion import ExposureFusion

fuser = ExposureFusion()

```

### Prerequisites

The class requires a set of common scientific and computer vision packages, such as:

* `numpy`
* `opencv`

For the rest of the files in the repository, one might also need:

* `tkinter`
* `matplotlib`
* `seaborn`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

As shown in the <a href="#getting-started">Getting Started</a> section, the class is easy to set up. 
Each method has an associated docstring that goes into its intended purpose and usage, however, one does *not* need to concern himself with the implementation details, since the class implements the `__call__` dunder method, that correctly performs the Exposure Fusion pipeline on the images

Example:

```python

from exposure_fusion import ExposureFusion

fuser = ExposureFusion(perform_alignment = False)

# Load images

images = [cv2.imread(path) for path in image_paths]

HDR = fuser(images)

```

The fusion pipeline's parameters are set by the class's `__init__` method, and can be changed by the user. The default parameters are what gave us the best result during our tests.



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Fix pyramid depth overexposure.

See the [open issues](https://github.com/dario-loi/exposure-fusion/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Dario Loi - dloi.projects@gmail.com

Project Link: [https://github.com/dario-loi/exposure-fusion](https://github.com/dario-loi/exposure-fusion)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [arpesenti](https://github.com/arpesenti/exposure_fusion) the author of an old implementation of the algorithm under the GPL-2 license.
* [Stanford Unversity](https://web.stanford.edu/class/cs231m/project-1/exposure-fusion.pdf) for the original paper.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/dario-loi/exposure-fusion.svg?style=for-the-badge
[contributors-url]: https://github.com/dario-loi/exposure-fusion/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/dario-loi/exposure-fusion.svg?style=for-the-badge
[forks-url]: https://github.com/dario-loi/exposure-fusion/network/members
[stars-shield]: https://img.shields.io/github/stars/dario-loi/exposure-fusion.svg?style=for-the-badge
[stars-url]: https://github.com/dario-loi/exposure-fusion/stargazers
[issues-shield]: https://img.shields.io/github/issues/dario-loi/exposure-fusion.svg?style=for-the-badge
[issues-url]: https://github.com/dario-loi/exposure-fusion/issues
[license-shield]: https://img.shields.io/github/license/dario-loi/exposure-fusion.svg?style=for-the-badge
[license-url]: https://github.com/dario-loi/exposure-fusion/blob/master/LICENSE.md
[product-screenshot]: images/HDR-output.png

[Python.org]: https://img.shields.io/github/pipenv/locked/python-version/dario-loi/exposure-fusion/master?style=for-the-badge
[Python-url]: https://www.python.org/