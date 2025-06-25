- **archivator.py:**  
  Splits files from an input directory into multiple zip archives, each not exceeding a specified maximum size. Useful for managing and distributing large datasets.

- **autopost_telegram_api.py:**  
  Automates posting images and captions to Telegram channels using the Telegram API, tracks sent images to prevent duplicates, and supports optional description files for captions.

- **config_for_tgmd.yaml:**  
  YAML configuration file containing credentials and settings for the telegram_media_downloader tool, such as API keys, download parameters, and channel lists.

- **lama.tar.xz:**  
  Compressed archive containing data for LaMa neural network, used for image inpainting and object removal tasks.

- **main.py:**  
  Pipeline for automated media collection, watermark segmentation, mask inference, and batch inpainting. Downloads images and videos, processes them with a segmentation model, prepares masks, runs LaMa inpainting, and applies postprocessing. Supports multi-threaded operations for large-scale processing.

- **requirements.txt:**  
  Lists all Python package dependencies required to run the core functionality and scripts of the project.

- **telegram_media_downloader.py:**  
  Downloads media from restricted Telegram channels using the Telegram API, supports batch download and configuration via YAML, useful for collecting datasets or archives from Telegram sources.

- **watermark_gen_2.py:**  
  Generates synthetic watermark overlays on input images, allowing random placement and realistic mask creation for segmentation training.

- **watermark_generation.py:**  
  Creates watermark overlays with colored backgrounds on input images, producing both watermarked images and corresponding binary masks for training segmentation models.

- **README.md:**  
  Do not read him.
