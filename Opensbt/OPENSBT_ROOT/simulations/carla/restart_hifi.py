import docker
import os

PATH_LOCAL = "/home/user/Documents/testing/lofi-hifi-praktikum-deployments/"

def restart_hifi():
    try:
        client = docker.from_env()

        container_name = "autoware_mini_hifi"

        # Define build args and environment variables
        build_args = {
            "PROCESS_COUNT": os.getenv("PROCESS_COUNT", "1"),
            "DISPLAY": os.getenv("DISPLAY", ":0"),
            "QT_X11_NO_MITSHM": "1",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "XAUTHORITY": "/tmp/.docker.xauth"
        }

        environment = [
            f"PYTHONPATH=/opt/CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:/opt/CARLA_ROOT/PythonAPI/carla/agents:/opt/CARLA_ROOT/PythonAPI/carla",
            f"CARLA_ROOT=/opt/CARLA_ROOT",
            f"DISPLAY={os.getenv('DISPLAY', ':0')}",
            "_NV_PRIME_RENDER_OFFLOAD=1",
            "GLX_VENDOR_LIBRARY_NAME=nvidia",
            "NVIDIA_VISIBLE_DEVICES=all",
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
            "ROS_MASTER_URI=http://localhost:11312"
        ]

        volumes = {
            "/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"},
            os.path.expanduser("~/.Xauthority"): {"bind": "/root/.Xauthority", "mode": "rw"},
            os.path.abspath(f"{PATH_LOCAL}/autoware_mini/SCENARIO_RUNNER"): {"bind": "/opt/SCENARIO_RUNNER", "mode": "rw"},
            os.path.abspath(f"{PATH_LOCAL}/autoware_mini/AUTOWARE_MINI"): {"bind": "/opt/catkin_ws/src/autoware_mini", "mode": "rw"},
        }

        image_tag = "autoware_mini:latest"
        dockerfile_rel_path = "AUTOWARE_MINI/Dockerfile"
        context_path = os.path.join(os.path.dirname(__file__), "autoware_mini")

        try:
            # Stop and remove the container if it exists
            try:
                container = client.containers.get(container_name)
                container.kill()
                print(f"Stopped container: {container_name}")
            except docker.errors.NotFound:
                print(f"Container {container_name} not found for killing")
            try:
                container = client.containers.get(container_name)
                container.remove(force=True)
                print(f"Removed container: {container_name}")
            except docker.errors.NotFound:
                print(f"Container {container_name} not found for removing")

            try:
                image = client.images.get(image_tag)
                print(f"Image '{image_tag}' already exists. Skipping build.")
            except docker.errors.ImageNotFound:
                print(f"Image '{image_tag}' not found. Building it now...")
                image, _ = client.images.build(
                    path=context_path,
                    dockerfile=dockerfile_rel_path,
                    tag=image_tag,
                    buildargs=build_args
                )
                print(f"Image '{image_tag}' built successfully.")
        except docker.errors.APIError as e:
            print(f"Docker API error: {e}")
            raise
        # Run container
        new_container = client.containers.run(
            image="autoware_mini:latest",
            name=container_name,
            command=[
                "bash", "-c",
                "source /opt/catkin_ws/devel/setup.bash && roslaunch autoware_mini start_carla.launch detector:=lidar_cluster use_scenario_runner:=true map_name:=Town01"
            ],
            stdin_open=True,
            tty=True,
            detach=True,
            environment=environment,
            volumes=volumes,
            network_mode="host",
            runtime="nvidia"
        )

        print(f"Container {container_name} started.")

    except Exception as e:
        print(f"Error: {e}")
