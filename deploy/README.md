# How to Deploy pi0

## 1. Build the Docker Image
```bash
./BUILD-DOCKER-CONTAINER.sh
```

## 2. Place the model weights into a directory (e.g., checkpoints)

## 3. Launch the Docker Container
```bash
./RUN-DOCKER-CONTAINER.sh
```

## 4. Inside the container, edit the following file to resolve the stack issue in rospy’s init:
/opt/ros/noetic/lib/python3/dist-packages/rosgraph/roslogging.py

Change this block:
```python
 while hasattr(f, "f_code"):
            # Search for the right frame using the data already found by parent class.
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == file_name and f.f_lineno == lineno and co.co_name == func_name:
                break
            if f.f_back:
                f = f.f_back
```
To this (just add one break):
```python
 while hasattr(f, "f_code"):
            # Search for the right frame using the data already found by parent class.
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == file_name and f.f_lineno == lineno and co.co_name == func_name:
                break
            if f.f_back:
                f = f.f_back
                break
```
Reference: https://github.com/ros/ros_comm/issues/2296

5. Run the model inference
```bash
roslaunch hsr_openpi hsr_openpi.launch \
	config_name:=pi0_hsr_weblab_leader \
	checkpoint_dir:=/home/openpi/checkpoints/pi0_hsr_weblab_leader/3000
```
To change the language instruction at runtime:
```bash
rosservice call /hsr_openpi/update_instruction "message: 'Open the oven toaster'"
```