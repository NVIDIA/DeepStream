# DeepStream Host Provisioning
Ansible playbooks for provisioning Ubuntu hosts with the NVIDIA driver, CUDA, cuDNN, TensorRT, NVIDIA Docker, and DeepStream dependencies. Supported platforms: **x86_64** & **ARM SBSA**. For Jetson configuration, use [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) with a supported JetPack version.

### Setup Details

Packages | Status | Versions (default values shown; all configurable)
 --- | --- | ---
OS Supported | **Ubuntu-24** | 
NVIDIA Device Driver (Open) | *CONFIGURABLE* | **nv_driver_version=** 595.45.04
CUDA | *CONFIGURABLE* | **cuda_version=** 13-2
CuDNN    | *CONFIGURABLE* | **cudnn_version=** 9  **cudnn_sub_version=** 9.20
TensorRT | *CONFIGURABLE* | **trt_version=** 10  **trt_sub_version=** 10.16
Nvidia Docker | **ADDED** | 

## Running the Playbook

### Prerequisites
- Ansible installed on your control machine (where you'll run the playbook from)
  ```bash
  sudo apt update
  sudo apt install ansible -y
  ```
- SSH access to the target machine(s)
- Target machine running Ubuntu 24.04

### Local Installation
1. Clone this repository:
   ```bash
   git clone <REPO_URL>
   cd <REPO_DIR>/tools/deepstream-host-provisioning

2. Create an inventory file (e.g., `inventory.ini`):
   ```ini
   [target]
   your-server-ip ansible_user=your-username
   ```

3. Run the playbook:
   ```bash
   ansible-playbook -i inventory.ini ubuntu_setup.yml
   ```

   Note: Adjust the version numbers according to your needs. The values shown above are examples.

### Available Variables
All variables are optional and have default values. If not specified, the following default values will be used:

- `nv_driver_version`: NVIDIA driver version (default: "595.45.04")
- `cuda_version`: CUDA version (format: XX-X) (default: "13-2")
- `cudnn_version`: cuDNN major version (default: "9")
- `cudnn_sub_version`: cuDNN sub-version (default: "9.20")
- `trt_version`: TensorRT major version (default: "10")
- `trt_sub_version`: TensorRT sub-version (default: "10.16")

You can override any of these values by passing them as extra variables (-e) when running the playbook.

### Basic Usage

#### Local System Configuration
The repository includes a default inventory file (`inventory.ini`) configured for local system setup. To configure your local system:

```bash
ansible-playbook -i inventory.ini ubuntu_setup.yml
```

This will use all default values and configure the local system. The playbook will run with elevated privileges (sudo) as required.

#### Custom Configuration
To override specific values while configuring local system:
```bash
ansible-playbook -i inventory.ini ubuntu_setup.yml \
  -e "cuda_version=13-2" \
  -e "nv_driver_version=595.45.04"
```

#### Remote System Configuration
For configuring remote systems, modify the inventory.ini file with your target host details:
```ini
[target]
your-server-ip ansible_user=your-username
```
