download_obb_v1:
	# mkdir obb_v1
	# curl -L "https://universe.roboflow.com/ds/z0mnVxh24L?key=WBcvU896CR" > obb_v1/roboflow.zip;
	# unzip obb_v1/roboflow.zip -d obb_v1/
	mv obb_v1/test/images/* obb_v1/train/images/
	mv obb_v1/test/labels/* obb_v1/train/labels/

download_segment_v5_2:
	mkdir segment_v5_2
	curl -L "https://universe.roboflow.com/ds/z1A5GQH1RJ?key=u5c9T0EDtB" > segment_v5_2/roboflow.zip;
	unzip segment_v5_2/roboflow.zip -d segment_v5_2/
	mv segment_v5_2/test/images/* segment_v5_2/train/images/
	mv segment_v5_2/test/labels/* segment_v5_2/train/labels/

build:
	python3 -m venv venv
	venv/bin/pip3 install -r requirements.txt

cuda:
	wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
	sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
	wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb
	sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb
	sudo cp /var/cuda-repo-wsl-ubuntu-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
	sudo apt-get update
	sudo apt-get -y install cuda-toolkit-12-3
